#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube ACR Recordings → Whisper → Firestore
Drive 変更をポーリングし、その場で音声または JSON を処理して Firestore に保存。
"""

import json
import os
import io
import logging
import tempfile
import uuid
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any

from flask import Flask, request, jsonify, abort
from google.cloud import firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import openai

# ──────────────────── 環境変数 ────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FOLDER_ID      = os.environ["FOLDER_ID"]                # Drive フォルダ ID
SA_INFO        = json.loads(os.environ["DRIVE_SA_KEY_JSON"])

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
db            = firestore.Client()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# JSON と AMR も許可
SUPPORTED_EXT = {
    "flac","m4a","mp3","mp4","mpeg","mpga",
    "oga","ogg","wav","webm","amr","json"
}

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
    """
    Drive から DL → 必要なら AMR→MP3 変換 → (path, ext) を返す
    """
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

    # AMR は MP3 に変換して Whisper に投げる
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
    """Whisper‐1 で文字起こし"""
    with open(src_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(
            file=f, model="whisper-1"
        )
    return resp.text

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
              "nextPageToken,"
              "newStartPageToken,"
              "changes(file(id,name,mimeType,parents))"
            )
        ).execute()

        for ch in resp.get("changes", []):
            file = ch.get("file") or {}
            mime = file.get("mimeType", "")
            # audio/ か JSON
            if not (mime.startswith("audio/") or file.get("name","").lower().endswith(".json")):
                continue
            # フォルダ親チェック
            if FOLDER_ID not in (file.get("parents") or []):
                continue

            fid, name = file["id"], file["name"]
            try:
                local, ext = download_from_drive(fid, name)

                if ext == "json":
                    # JSON なら直接読み込み
                    data = json.load(open(local, encoding="utf-8"))
                    text = data.get("text", data.get("transcript",""))
                else:
                    # audio → Whisper
                    text = transcription(local)

                enqueue_record({
                    "fileId"   : fid,
                    "fileName" : name,
                    "text"     : text,
                    "status"   : "done",
                    "createdAt": firestore.SERVER_TIMESTAMP,
                })
                polled += 1
                logging.info("✅ success: %s", name)
            except Exception:
                logging.exception("❌ failed: %s", name)

        token = (
            resp.get("newStartPageToken")
            or resp.get("nextPageToken")
            or token
        )
        if not resp.get("nextPageToken"):
            break

    meta_ref.set({"value": token})
    logging.info("polled=%s", polled)
    return jsonify({"polled": polled})

@app.post("/webhook")
def webhook():
    data    = request.get_json(force=True)
    fid     = data["fileId"]
    name    = data["fileName"]
    logging.info("▶️ start webhook: %s %s", fid, name)

    try:
        local, ext = download_from_drive(fid, name)
        if ext == "json":
            data = json.load(open(local, encoding="utf-8"))
            text = data.get("text", data.get("transcript",""))
        else:
            text = transcription(local)

        enqueue_record({
            "fileId"   : fid,
            "fileName" : name,
            "text"     : text,
            "status"   : "done",
            "createdAt": firestore.SERVER_TIMESTAMP,
        })
        logging.info("✅ success")
        return jsonify({"ok": True})
    except Exception:
        logging.exception("❌ webhook failed")
        abort(500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")), debug=False)
