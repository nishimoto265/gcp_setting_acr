#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube ACR Recordings → Whisper → Firestore & Firebase Storage
Drive の変更をポーリングし、新規音声ファイルをダウンロード → 文字起こし →
日付・電話番号抽出 → LangChain で要約 → Firebase Storage に音声保存 →
Firestore にメタデータ保存
"""

import os, io, json, logging, tempfile, uuid, re, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Any, Dict, Optional

from flask import Flask, request, jsonify, abort
from google.cloud import firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import openai

import firebase_admin
from firebase_admin import credentials as fb_credentials, storage as fb_storage

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from flask_cors import CORS

app = Flask(__name__)
# localhost からの開発アクセスだけ許可
CORS(app,
     origins=["http://localhost:8083"],
     methods=["POST", "OPTIONS"],
     allow_headers=["Content-Type", "Origin"])



# ──────────────────── 環境変数 ────────────────────
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY      = os.environ["GOOGLE_API_KEY"]
SUMMARY_MODEL       = os.getenv("SUMMARY_MODEL", "gemini-1.5-flash-latest")

DRIVE_FOLDER_ID     = os.environ["FOLDER_ID"]
DRIVE_SA_KEY_JSON   = json.loads(os.environ["DRIVE_SA_KEY_JSON"])
FIREBASE_SA_KEY     = json.loads(os.environ["FIREBASE_SA_KEY_JSON"])
FIREBASE_BUCKET     = os.getenv("STORAGE_BUCKET", "nodal-alcove-457508-h6.appspot.com")
FIREBASE_PROJECT_ID = "nodal-alcove-457508-h6"


# ──────────────────── GCP/Firebase 初期化 ────────────────────
gcp_creds = service_account.Credentials.from_service_account_info(FIREBASE_SA_KEY)
db        = firestore.Client(project=FIREBASE_PROJECT_ID, credentials=gcp_creds)

# ★ Whisper API につなぐ OpenAI クライアントに 10 分のタイムアウトを付ける
openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=600   # seconds
)

cred = fb_credentials.Certificate(FIREBASE_SA_KEY)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
bucket = fb_storage.bucket()


# ──────────────────── LangChain 要約チェーン ────────────────────
chat_llm = ChatGoogleGenerativeAI(
    model=SUMMARY_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    convert_system_message_to_human=True,
)
prompt = PromptTemplate(
    input_variables=["text"],
    template="以下の通話内容を日本語で 3 行以内で要約してください：\n\n{text}",
)
summarize_chain = LLMChain(llm=chat_llm, prompt=prompt)


# ──────────────────── アプリ設定 ────────────────────
SUPPORTED_EXT = {
    "flac", "m4a", "mp3", "mp4", "mpeg", "mpga",
    "oga", "ogg", "wav", "webm", "amr"
}

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


# ──────────────────── 共通ユーティリティ ────────────────────
def get_drive() -> Any:
    creds = service_account.Credentials.from_service_account_info(
        DRIVE_SA_KEY_JSON,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        ],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def clean_ext(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")


def _maybe_shrink(path: str) -> str:
    """
    24 MB を超えるファイルは 16 kHz / 64 kbps モノラル MP3 へ再エンコード。
    Whisper へのアップロードタイムアウト回避用。
    """
    if os.path.getsize(path) <= 24_000_000:
        return path

    shrunk = str(Path(path).with_suffix(".mp3"))
    subprocess.run(
        ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-b:a", "64k", "-y", shrunk],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return shrunk


def download_from_drive(file_id: str, name: str) -> Tuple[str, str]:
    """Drive からファイルをローカルに保存し、必要に応じて変換も行う"""
    ext = clean_ext(name)
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"unsupported ext: {ext}")

    drive = get_drive()
    req   = drive.files().get_media(fileId=file_id)
    buf   = io.BytesIO()
    dl    = MediaIoBaseDownload(buf, req)
    done  = False
    while not done:
        _, done = dl.next_chunk()

    tempdir  = tempfile.mkdtemp()
    original = os.path.join(tempdir, f"{uuid.uuid4()}.{ext}")
    with open(original, "wb") as out_f:
        out_f.write(buf.getvalue())

    # AMR → MP3
    if ext == "amr":
        converted = os.path.join(tempdir, f"{uuid.uuid4()}.mp3")
        subprocess.run(
            ["ffmpeg", "-i", original, "-acodec", "libmp3lame", "-y", converted],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
        )
        original, ext = converted, "mp3"

    # 大容量なら圧縮
    original = _maybe_shrink(original)
    ext      = clean_ext(original)

    return original, ext


def upload_to_firebase(local_path: str, dest_path: str) -> str:
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_path)

    # Uniform bucket-level access が有効でも署名 URL は使える
    expires = datetime.utcnow() + timedelta(days=7)
    return blob.generate_signed_url(expires)


def transcription(src_path: str) -> str:
    """Whisper に 3 回までリトライで投げる"""
    for i in range(3):
        try:
            with open(src_path, "rb") as f:
                r = openai_client.audio.transcriptions.create(
                    file=f, model="whisper-1", language="ja"
                )
            return r.text
        except Exception as e:
            logging.warning(f"Whisper retry {i+1}/3… {e}")
            time.sleep(2 ** i)   # 1s → 2s → 4s
    raise RuntimeError("Whisper failed after 3 retries")


def summarize(text: str) -> str:
    result = summarize_chain.invoke({"text": text})
    return result["text"] if isinstance(result, dict) else result


# 電話番号 / 日時抽出
_PHONE_PAT = re.compile(
    r"""
        (?:
            0\d{1,4}-\d{1,4}-\d{3,4} |  # 03-1234-5678, 098-123-4567 …
            0\d{9,10} |                # 09012345678 …
            0570-\d{3}-\d{3}           # 0570-xxx-xxx
        )
    """, re.VERBOSE
)
_DATE_PAT  = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})")


def parse_filename(name: str) -> Tuple[Optional[str], datetime]:
    m_phone = _PHONE_PAT.search(name)
    phone   = m_phone.group(0) if m_phone else None

    m_date  = _DATE_PAT.search(name)
    if not m_date:
        raise ValueError(f"Datetime not found in filename: {name}")
    recorded_at = datetime.strptime(m_date.group(1), "%Y-%m-%d %H-%M-%S")
    return phone, recorded_at


def enqueue_record(file_id: str, meta: Dict[str, Any]) -> None:
    ref = db.collection("recordings").document(file_id)
    if ref.get().exists:
        logging.info(f"⏭ skip existing record: {file_id}")
        return
    ref.set(meta)


# ──────────────────── 共通処理本体 ────────────────────
def _process_file(fid: str, name: str) -> None:
    local, ext     = download_from_drive(fid, name)
    phone, rec_at  = parse_filename(name)
    dest           = f"recordings/{rec_at.date()}/{uuid.uuid4()}.{ext}"
    audio_url      = upload_to_firebase(local, dest)
    text           = transcription(local)
    summary        = summarize(text)

    meta = {
        "fileId":      fid,
        "fileName":    name,
        "phoneNumber": phone,
        "recordedAt":  rec_at,
        "transcript":  text,
        "summary":     summary,
        "audioUrl":    audio_url,
        "status":      "done",
        "createdAt":   firestore.SERVER_TIMESTAMP,
    }
    enqueue_record(fid, meta)
    logging.info(f"✅ processed: {name} (ID: {fid})")


# ──────────────────── Flask エンドポイント ────────────────────
@app.post("/process_all")
def process_all():
    drive    = get_drive()
    existing = {d.id for d in db.collection("recordings").stream()}

    resp = drive.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType contains 'audio/'",
        fields="files(id,name)"
    ).execute()

    todo = [f for f in resp.get("files", []) if f["id"] not in existing]
    done = 0
    for f in todo:
        try:
            _process_file(f["id"], f["name"])
            done += 1
        except Exception:
            logging.exception(f"❌ failed: {f['name']}")

    return jsonify({"to_process": len(todo), "processed": done})


@app.post("/poll")
def poll():
    body      = request.get_json(force=True, silent=True) or {}
    drive     = get_drive()
    meta_ref  = db.collection("meta").document("drivePageToken")
    token_doc = meta_ref.get()
    token     = body.get("startPageToken") or (token_doc.to_dict() or {}).get("value")
    if not token:
        token = drive.changes().getStartPageToken().execute()["startPageToken"]

    polled = 0
    while True:
        resp = drive.changes().list(
            pageToken=token, spaces="drive",
            fields="nextPageToken,newStartPageToken,"
                   "changes(file(id,name,mimeType,parents))"
        ).execute()

        for ch in resp.get("changes", []):
            f = ch.get("file") or {}
            if not (
                f.get("mimeType", "").startswith("audio/")
                and DRIVE_FOLDER_ID in (f.get("parents") or [])
            ):
                continue
            try:
                _process_file(f["id"], f["name"])
                polled += 1
            except Exception:
                logging.exception(f"❌ failed: {f.get('name')}")

        token = resp.get("newStartPageToken") or resp.get("nextPageToken") or token
        if "nextPageToken" not in resp:
            break

    meta_ref.set({"value": token})
    return jsonify({"polled": polled})


@app.post("/webhook")
def webhook():
    data = request.get_json(force=True)
    try:
        _process_file(data["fileId"], data["fileName"])
        return jsonify({"ok": True})
    except Exception:
        logging.exception(f"❌ webhook failed: {data.get('fileName')}")
        abort(500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
