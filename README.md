# Cube ACR 録音文字起こしサービス

Google Drive フォルダを監視し、音声ファイルや JSON ファイルを取得して文字起こしを行い、Firest​ore に結果を保存するサーバーレスアプリケーションです。

- **Drive 監視**：Cloud Scheduler で毎分 `/poll` エンドポイントを呼び出し、新規ファイルを検出
- **音声処理**：`.m4a`、`.mp3`、`.wav`、`.amr` などをダウンロード、`.amr` は FFmpeg で MP3 に変換
- **文字起こし**：OpenAI Whisper API で音声を文字起こし、または事前生成された `.json` をそのまま読み込む
- **データ保存**：Firestore の `recordings` コレクションにメタ情報と文字起こしテキストを保存
- **トークン管理**：Firestore の `meta/drivePageToken` ドキュメントにページトークンを保持し、毎回「続き」から処理

---

## 主な機能

- Drive フォルダの変更検知（追加のみ）
- 複数フォーマット対応（音声 + JSON）
- エラー回避のための拡張子チェック & AMR→MP3 変換
- 動的なトークン管理により重複処理を防止

---

## 前提環境

1. **GCP プロジェクト**（請求有効化済み）
2. **サービスアカウント**（`drive-watcher@…`）に以下の権限を付与
   - `roles/iam.serviceAccountUser`
   - `roles/run.invoker`
   - `roles/datastore.user`
   - `roles/secretmanager.secretAccessor`（Secret Manager 参照用）
3. **Secret Manager** に Drive 用サービスアカウント JSON キー (`DRIVE_SA_KEY_JSON`) を登録
4. **OpenAI API Key**（Whisper 利用可能なキー）
5. 監視対象の **Drive フォルダ ID** を控える

---

## 環境変数

| 変数名               | 説明                                         |
|----------------------|----------------------------------------------|
| `OPENAI_API_KEY`     | OpenAI API キー                              |
| `FOLDER_ID`          | Google Drive のフォルダ ID                   |
| `DRIVE_SA_KEY_JSON`  | Secret Manager から注入したサービスアカウントJSON |
| `PORT`               | Flask がリッスンするポート（省略時: 8080）    |

---

## ローカル開発

```bash
# リポジトリをクローン
git clone https://github.com/nishimoto265/gcp_setting_acr.git
cd gcp_setting_acr

# 依存パッケージをインストール
pip install -r requirements.txt

# 環境変数を設定
export OPENAI_API_KEY=<あなたのキー>
export FOLDER_ID=<フォルダID>
export DRIVE_SA_KEY_JSON="$(< service-account.json)"

# ローカル起動
flask run --port=8080
```

---

## Cloud Run へのデプロイ

```bash
# プロジェクトディレクトリへ
cd ~/project

# 変更をコミット
git add main.py Dockerfile
git commit -m "feat: <変更内容>"

# 実際のフォルダIDをセット
REAL_ID=<実際のフォルダID>

# Cloud Run にデプロイ
gcloud run deploy acr-3 \
  --source . \
  --region=asia-northeast2 \
  --clear-base-image \
  --service-account=acr-runner@${PROJECT_ID}.iam.gserviceaccount.com \
  --set-env-vars="FOLDER_ID=${REAL_ID},OPENAI_API_KEY=<あなたのキー>,DRIVE_SA_KEY_JSON=<シークレット参照>"
```

---

## Cloud Scheduler 設定

- **ジョブ ID**: `drive-poll-job`
- **スケジュール**: `* * * * *`（毎分）
- **HTTP メソッド**: POST
- **URL**: `https://<Service_URL>/poll`
- **ボディ**: `{}`

---

## Firestore 構成

### recordings コレクション
- `fileId` (string)
- `fileName` (string)
- `text` (string)
- `status` ("done" など)
- `createdAt` (timestamp)

### meta コレクション
- ドキュメント: `drivePageToken`
  - フィールド: `value` (string)

---

## トラブルシューティング

- **`polled=0`**
  - `FOLDER_ID` が正しいか、SA にフォルダ権限があるか
  - Secret Manager のトークンリセット（Firestore から `drivePageToken` 削除）
- **Invalid file format**
  - `.amr`, `.json` が許可リストに入っているか
  - Dockerfile に FFmpeg がインストールされているか
- **ログ確認**

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="acr-3"' \
  --limit 20 --format='table(timestamp,textPayload)'
```

---

## ライセンス

MIT License

