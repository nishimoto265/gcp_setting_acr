import os
import uuid
import json
import google.auth
from googleapiclient.discovery import build

# ─── 環境変数 ─────────────────────────────────
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

def init_watch():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive = build("drive", "v3", credentials=creds)

    # 1) startPageToken 取得＆永続化
    resp = drive.changes().getStartPageToken().execute()
    token = resp["startPageToken"]
    with open("page_token.txt", "w") as f:
        f.write(token)

    # 2) チャネル登録
    channel_id = uuid.uuid4().hex
    body = {
        "id":      channel_id,
        "type":    "webhook",
        "address": WEBHOOK_URL
    }
    channel = drive.changes().watch(
        pageToken=token,
        body=body
    ).execute()

    print("▶ Watch channel created:")
    print(json.dumps(channel, indent=2))
    # 必要なら channel["resourceId"]・["expiration"] などを保存

if __name__ == "__main__":
    init_watch()
