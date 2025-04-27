#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube ACR Recordings → Whisper → Firestore
Drive 変更をポーリングし、その場で音声を文字起こし → 日付・電話番号抽出 → 要約生成 → Firestoreに保存
"""

import os
import io
import json
import logging
import tempfile
import uuid
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Dict

from flask import Flask, request, jsonify, abort
from google.cloud import firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import openai

# LangChain import
from langchain import OpenAI as LcOpenAI, PromptTemplate, LLMChain

# ──────────────────── 環境変数 ────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FOLDER_ID      = os.environ["FOLDER_ID"]
SA_INFO        = json.loads(os.environ["DRIVE_SA_KEY_JSON"])

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
db            = firestore.Client()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# LangChain 要約チェーン設定
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="以下の通話内容を日本語で簡潔に要約してください：\n\n{text}"
)
llm = LcOpenAI(model_name="gpt-4o-mini", temperature=0.5)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# 音声のみ許可
SUPPORTED_EXT = {"flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm","amr"}


def get_drive() -> Any:
    creds = service_account.Credentials.from_service_account_info(
        SA_INFO,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        ],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def clean_ext(name: str) -> str:
    return Path(name).suffix.lower().lstrip(" .")


def download_from_drive(file_id: str, name: str) -> Tuple[str, str]:
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

    tempdir = tempfile.mkdtemp()
    original = os.path.join(tempdir, f"{uuid.uuid4()}.{ext}")
    with open(original, "wb") as out_f:
        out_f.write(buf.getvalue())

    if ext == "amr":
        converted = os.path.join(tempdir, f"{uuid.uuid4()}.mp3")
        subprocess.run(
            ["ffmpeg", "-i", original, "-acodec", "mp3", "-y", converted],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return converted, "mp3"

    return original, ext


def transcription(src_path: str) -> str:
    """Whisper-1 で文字起こし（日本語固定）"""
    with open(src_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            language="ja"
        )
    return resp.text


def summarize(text: str) -> str:
    """LangChain を使った日本語要約"""
    return summarize_chain.run(text)


def parse_filename(name: str) -> Tuple[str, datetime]:
    """ファイル名から電話番号と録音日時を抽出"""
    m_phone = re.match(r"^([\d-]+)", name)
    if not m_phone:
        raise ValueError(f"Phone number not found in filename: {name}")
    phone = m_phone.group(1)

    m_date = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})", name)
    if not m_date:
        raise ValueError(f"Datetime not found in filename: {name}")
    recorded_at = datetime.strptime(m_date.group(1), "%Y-%m-%d %H-%M-%S")

    return phone, recorded_at


def enqueue_record(meta: Dict[str, Any]) -> None:
    db.collection("recordings").add(meta)


@app.post("/poll")
def poll():
    body   = request.get_json(force=True, silent=True) or {}
    drive  = get_drive()

    meta_ref  = db.collection("meta").document("drivePageToken")
    token_doc = meta_ref.get()
    token     = body.get("startPageToken") or (token_doc.to_dict() or {}).get("value")
    if not token:
        token = drive.changes().getStartPageToken().execute()["startPageToken"]

    polled = 0
    while True:
        resp = drive.changes().list(
            pageToken=token,
            spaces="drive",
            fields=(
                "nextPageToken,newStartPageToken,changes(file(id,name,mimeType,parents))"
            )
        ).execute()

        for ch in resp.get("changes", []):
            file = ch.get("file") or {}
            mime = file.get("mimeType", "")
            if not mime.startswith("audio/"):
                continue
            if FOLDER_ID not in (file.get("parents") or []):
                continue

            fid, name = file["id"], file["name"]
            try:
                local, ext = download_from_drive(fid, name)
                text = transcription(local)
                phone, recorded_at = parse_filename(name)
                summary = summarize(text)

                enqueue_record({
                    "fileId": fid,
                    "fileName": name,
                    "phoneNumber": phone,
                    "recordedAt": recorded_at,
                    "transcript": text,
                    "summary": summary,
                    "status": "done",
                    "createdAt": firestore.SERVER_TIMESTAMP
                })
                polled += 1
                logging.info("✅ success: %s", name)
            except Exception:
                logging.exception("❌ failed: %s", name)

        token = resp.get("newStartPageToken") or resp.get("nextPageToken") or token
        if not resp.get("nextPageToken"):
            break

    meta_ref.set({"value": token})
    logging.info("polled=%s", polled)
    return jsonify({"polled": polled})


@app.post("/webhook")
def webhook():
    data = request.get_json(force=True)
    fid  = data["fileId"]
    name = data["fileName"]
    logging.info("▶️ start webhook: %s %s", fid, name)

    try:
        local, ext = download_from_drive(fid, name)
        text = transcription(local)
        phone, recorded_at = parse_filename(name)
        summary = summarize(text)

        enqueue_record({
            "fileId": fid,
            "fileName": name,
                    "phoneNumber": phone,
                    "recordedAt": recorded_at,
                    "transcript": text,
                    "summary": summary,
                    "status": "done",
                    "createdAt": firestore.SERVER_TIMESTAMP
                })
        logging.info("✅ success")
        return jsonify({"ok": True})
    except Exception:
        logging.exception("❌ webhook failed")
        abort(500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
