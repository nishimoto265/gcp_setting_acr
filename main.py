#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube ACR Work-Flow

 recording/   : 新規音声 → Whisper 文字起こし → Gemini 1.5 Flash 要約 → Firestore(recordings)
 screenshot/ : 新規画像 → 前処理 → (Document AI ⇒ Vision OCR) → 正規表現抽出
                → Gemini 1.5 Flash 補完 → Firestore(tickets)
"""

# ───────────────────────── Imports ─────────────────────────
import os, io, json, logging, tempfile, uuid, re, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Any, Dict, Optional

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# Google APIs
from google.cloud import firestore, vision
from google.cloud import documentai_v1 as documentai   # ← ここだけ分離
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Firebase
import firebase_admin
from firebase_admin import credentials as fb_credentials, storage as fb_storage

# OpenAI Whisper
import openai

# LangChain / Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 画像前処理
import cv2
import numpy as np

# ───────────────────────── Flask ─────────────────────────
app = Flask(__name__)
CORS(app,
     origins=["http://localhost:8083"],
     methods=["POST", "OPTIONS"],
     allow_headers=["Content-Type", "Origin"])

# ───────────────────────── ENV ─────────────────────────
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY      = os.environ["GOOGLE_API_KEY"]
SUMMARY_MODEL       = os.getenv("SUMMARY_MODEL", "gemini-1.5-flash-latest")

RECORDING_FOLDER_ID   = os.environ["FOLDER_ID"]            # recording サブフォルダ
SCREENSHOT_FOLDER_ID  = os.environ["SCREENSHOT_FOLDER_ID"] # screenshot サブフォルダ
DOC_AI_PROCESSOR      = os.getenv("DOC_AI_PROCESSOR")      # projects/…/processors/… 省略可

DRIVE_SA_KEY_JSON   = json.loads(os.environ["DRIVE_SA_KEY_JSON"])
FIREBASE_SA_KEY     = json.loads(os.environ["FIREBASE_SA_KEY_JSON"])
FIREBASE_BUCKET     = os.getenv("STORAGE_BUCKET", "nodal-alcove-457508-h6.appspot.com")
FIREBASE_PROJECT_ID = "nodal-alcove-457508-h6"

# ───────────────────────── GCP / Firebase ─────────────────────────
gcp_creds = service_account.Credentials.from_service_account_info(FIREBASE_SA_KEY)
db        = firestore.Client(project=FIREBASE_PROJECT_ID, credentials=gcp_creds)

cred = fb_credentials.Certificate(FIREBASE_SA_KEY)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
bucket = fb_storage.bucket()

# ───────────────────────── Clients ─────────────────────────
openai_client  = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=600)
vision_client  = vision.ImageAnnotatorClient(credentials=gcp_creds)
docai_client   = (documentai.DocumentProcessorServiceClient(credentials=gcp_creds)
                  if DOC_AI_PROCESSOR else None)

# ───────────────────────── Gemini LLMs ─────────────────────────
# ① Whisper 要約
chat_summarize = ChatGoogleGenerativeAI(
    model=SUMMARY_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    convert_system_message_to_human=True,
)
summarize_chain = LLMChain(
    llm=chat_summarize,
    prompt=PromptTemplate(
        input_variables=["text"],
        template="以下の通話内容を日本語で 3 行以内で要約してください：\n\n{text}",
    ),
)

# ② OCR → JSON 補完
field_list = [
    "reception_no", "urgency_level", "reception_date",
    "entrusted_content", "primary_responsible",
    "contact_phone", "contact_name", "visit_address",
    "item", "item_model", "serial_no", "split_no",
    "request_content", "internal_message", "client_message",
    "request_category", "client", "client_tel", "client_fax",
    "client_order_no", "billing_category", "billing_code"
]
chat_extract = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    convert_system_message_to_human=False,
)
extract_chain = LLMChain(
    llm=chat_extract,
    prompt=PromptTemplate(
        input_variables=["text", "fields"],
        template=(
            "以下はOCRテキストです。文脈から推測して "
            "{fields} の JSON を完成させてください。出力は JSON のみ:\n\n```{text}```"
        ),
    ),
)

def llm_extract(raw: str) -> Dict[str, Any]:
    try:
        out = extract_chain.invoke({"text": raw, "fields": field_list})
        payload = out["text"] if isinstance(out, dict) else out
        return json.loads(payload)
    except Exception as e:
        logging.warning(f"LLM JSON 抽出失敗: {e}")
        return {}

# ───────────────────────── 定数 ─────────────────────────
SUPPORTED_EXT = {"flac", "m4a", "mp3", "mp4", "mpeg", "mpga",
                 "oga", "ogg", "wav", "webm", "amr"}

LABEL_MAP = {  # OCR ラベル ↔ Firestore フィールド
    "受付No": "reception_no", "緊急度": "urgency_level", "受付日": "reception_date",
    "委託内容": "entrusted_content", "主担当": "primary_responsible",
    "訪問先電話番号": "contact_phone", "訪問先氏名": "contact_name",
    "訪問先住所": "visit_address", "品目": "item", "品番": "item_model",
    "製造No": "serial_no", "分割No": "split_no", "依頼内容": "request_content",
    "社内メッセージ": "internal_message", "依頼元メッセージ": "client_message",
    "依頼分類": "request_category", "依頼元": "client",
    "依頼元電話番号": "client_tel", "依頼元FAX": "client_fax",
    "依頼元注番": "client_order_no", "請求区分": "billing_category",
    "請求先CD": "billing_code",
}

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

# ───────────────────────── Drive ─────────────────────────
def get_drive() -> Any:
    creds = service_account.Credentials.from_service_account_info(
        DRIVE_SA_KEY_JSON,
        scopes=["https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/drive.metadata.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ───────────────────────── 画像前処理 ─────────────────────────
def preprocess(path: str) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(img == 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    scale = 1300 / w if w < 1300 else 1
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, img)
    return tmp.name

# ───────────────────────── Vision OCR ─────────────────────────
_KV_RE = re.compile(r'(.+?)[:：]\s*(.+)')

def vision_ocr_extract(path: str) -> Dict[str, Any]:
    raw_path = preprocess(path)
    with open(raw_path, "rb") as f:
        img = vision.Image(content=f.read())
    res = vision_client.text_detection(image=img)
    raw = res.full_text_annotation.text if res.text_annotations else ""
    data: Dict[str, Any] = {}
    for line in [l.strip() for l in raw.splitlines() if l.strip()]:
        m = _KV_RE.match(line)
        if not m:
            continue
        label, val = m.groups()
        field = LABEL_MAP.get(label)
        if field:
            if field == "reception_date":
                try:
                    val = (datetime.strptime(val.replace('.', '/'), "%Y/%m/%d")
                           .strftime("%Y-%m-%d"))
                except ValueError:
                    pass
            data[field] = val
    # メッセージ系（複数行）
    for header, fld in [("社内メッセージ", "internal_message"),
                        ("依頼元メッセージ", "client_message")]:
        if fld in data or header not in raw:
            continue
        start = raw.index(header) + len(header)
        rest  = raw[start:]
        next_hdrs = [h for h in LABEL_MAP if h in rest and h != header]
        end = min(rest.index(h) for h in next_hdrs) if next_hdrs else len(rest)
        data[fld] = rest[:end].strip("：:\n ")
    # LLM で補完
    data.update({k: v for k, v in llm_extract(raw).items() if v})
    return data

# ───────────────────────── Document AI ─────────────────────────
def docai_extract(path: str) -> Dict[str, Any]:
    if not DOC_AI_PROCESSOR:
        return {}
    with open(path, "rb") as f:
        raw = f.read()
    resp = docai_client.process_document(
        request={"name": DOC_AI_PROCESSOR,
                 "raw_document": {"content": raw, "mime_type": "image/png"}},
    )
    out: Dict[str, Any] = {}
    for ent in resp.document.entities:
        field = LABEL_MAP.get(ent.type_ or ent.entity_type)
        if field:
            out[field] = ent.mention_text
    return out

# ───────────────────────── Audio Utils ─────────────────────────
def clean_ext(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")

def _maybe_shrink(path: str) -> str:
    if os.path.getsize(path) <= 24_000_000:
        return path
    out = str(Path(path).with_suffix(".mp3"))
    subprocess.run(
        ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-y", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def download_audio(fid: str, name: str) -> Tuple[str, str]:
    ext = clean_ext(name)
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"unsupported ext: {ext}")
    drive = get_drive()
    buf, done = io.BytesIO(), False
    dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=fid))
    while not done:
        _, done = dl.next_chunk()
    tmpdir = tempfile.mkdtemp()
    local  = os.path.join(tmpdir, f"{uuid.uuid4()}.{ext}")
    with open(local, "wb") as f:
        f.write(buf.getvalue())
    if ext == "amr":
        converted = os.path.join(tmpdir, f"{uuid.uuid4()}.mp3")
        subprocess.run(
            ["ffmpeg", "-i", local, "-acodec", "libmp3lame", "-y", converted],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        local, ext = converted, "mp3"
    return _maybe_shrink(local), clean_ext(local)

def download_image(fid: str, name: str) -> str:
    """Drive から画像をローカルに保存してパスを返す"""
    drive = get_drive()
    buf, done = io.BytesIO(), False
    dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=fid))
    while not done:
        _, done = dl.next_chunk()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
    tmp.write(buf.getvalue()); tmp.close()
    return tmp.name


def upload_to_firebase(local: str, dest: str) -> str:
    blob = bucket.blob(dest)
    blob.upload_from_filename(local)
    return blob.generate_signed_url(datetime.utcnow() + timedelta(days=7))

def transcription(src: str) -> str:
    for i in range(3):
        try:
            with open(src, "rb") as f:
                r = openai_client.audio.transcriptions.create(
                    file=f, model="whisper-1", language="ja")
            return r.text
        except Exception as e:
            logging.warning(f"Whisper retry {i+1}/3: {e}")
            time.sleep(2 ** i)
    raise RuntimeError("Whisper failed 3 times")

def summarize(text: str) -> str:
    res = summarize_chain.invoke({"text": text})
    return res["text"] if isinstance(res, dict) else res

_PHONE_PAT = re.compile(r'(0\d{1,4}-\d{1,4}-\d{3,4}|0\d{9,10}|0570-\d{3}-\d{3})')
_DATE_PAT  = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})')

def parse_filename(name: str) -> Tuple[Optional[str], datetime]:
    phone = _PHONE_PAT.search(name)
    mdate = _DATE_PAT.search(name)
    if not mdate:
        raise ValueError(f"date not found: {name}")
    dt = datetime.strptime(mdate.group(1), "%Y-%m-%d %H-%M-%S")
    return phone.group(0) if phone else None, dt

# ───────────────────────── Firestore Helpers ─────────────────────────
def save_recording(fid: str, meta: Dict[str, Any]) -> None:
    ref = db.collection("recordings").document(fid)
    if not ref.get().exists:
        ref.set(meta)

def save_ticket(fid: str, name: str, fields: Dict[str, Any]) -> None:
    ref = db.collection("tickets").document(fid)
    if ref.get().exists:
        logging.info(f"⏭ skip ticket: {name}")
        return
    ts = firestore.SERVER_TIMESTAMP
    ref.set({"fileId": fid, "fileName": name, **fields,
             "status": "ocr_done", "createdAt": ts, "updatedAt": ts})

# ───────────────────────── Processors ─────────────────────────
def process_audio(fid: str, name: str) -> None:
    local, ext = download_audio(fid, name)
    phone, rec_at = parse_filename(name)
    url  = upload_to_firebase(local, f"recordings/{rec_at.date()}/{uuid.uuid4()}.{ext}")
    text = transcription(local)
    save_recording(fid, {
        "fileId": fid, "fileName": name,
        "phoneNumber": phone, "recordedAt": rec_at,
        "transcript": text, "summary": summarize(text),
        "audioUrl": url, "status": "done",
        "createdAt": firestore.SERVER_TIMESTAMP,
    })
    logging.info(f"✅ audio: {name}")

def process_image(fid: str, name: str) -> None:
    local = download_image(fid, name)
    fields = docai_extract(local) or vision_ocr_extract(local)
    save_ticket(fid, name, fields)
    logging.info(f"✅ image: {name}")

# ───────────────────────── Flask Endpoints ─────────────────────────
@app.post("/process_all")
def process_all():
    drive = get_drive()
    rec_processed = {d.id for d in db.collection("recordings").stream()}
    img_processed = {d.id for d in db.collection("tickets").stream()}

    rec_files = drive.files().list(
        q=f"'{RECORDING_FOLDER_ID}' in parents and mimeType contains 'audio/'",
        fields="files(id,name)").execute().get("files", [])
    img_files = drive.files().list(
        q=f"'{SCREENSHOT_FOLDER_ID}' in parents and mimeType contains 'image/'",
        fields="files(id,name)").execute().get("files", [])

    a=i=0
    for f in rec_files:
        if f["id"] not in rec_processed:
            try: process_audio(f["id"], f["name"]); a+=1
            except Exception: logging.exception("audio err")
    for f in img_files:
        if f["id"] not in img_processed:
            try: process_image(f["id"], f["name"]); i+=1
            except Exception: logging.exception("img err")

    return jsonify({"audio": a, "image": i})

@app.post("/poll")
def poll():
    body = request.get_json(force=True, silent=True) or {}
    drive    = get_drive()
    meta_ref = db.collection("meta").document("drivePageToken")
    token    = body.get("startPageToken") or (meta_ref.get().to_dict() or {}).get("value")
    if not token:
        token = drive.changes().getStartPageToken().execute()["startPageToken"]

    a=i=0
    while True:
        rsp = drive.changes().list(
            pageToken=token, spaces="drive",
            fields=("nextPageToken,newStartPageToken,"
                    "changes(file(id,name,mimeType,parents))")).execute()

        for ch in rsp.get("changes", []):
            f = ch.get("file") or {}; mime=f.get("mimeType","")
            pid = (f.get("parents") or [None])[0]
            try:
                if mime.startswith("audio/") and pid == RECORDING_FOLDER_ID:
                    process_audio(f["id"], f["name"]); a+=1
                elif mime.startswith("image/") and pid == SCREENSHOT_FOLDER_ID:
                    process_image(f["id"], f["name"]); i+=1
            except Exception: logging.exception("poll err")

        token = rsp.get("newStartPageToken") or rsp.get("nextPageToken") or token
        if "nextPageToken" not in rsp: break
    meta_ref.set({"value": token})
    return jsonify({"audio": a, "image": i})

@app.post("/webhook")
def webhook():
    data = request.get_json(force=True)
    try:
        if data["type"] == "audio":
            process_audio(data["fileId"], data["fileName"])
        elif data["type"] == "image":
            process_image(data["fileId"], data["fileName"])
        else:
            abort(400, "unknown type")
        return jsonify(ok=True)
    except Exception:
        logging.exception("webhook err"); abort(500)

# ───────────────────────── Entrypoint ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
