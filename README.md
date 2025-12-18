# FX-Kline

<div align="center">

**AI駆動の外国為替OHLCデータ取得ツール**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

[English](./README_EN.md) | 日本語

</div>

---

## 📖 目次

- [概要](#概要)
- [主要機能](#主要機能)
- [デモ](#デモ)
- [対応マーケット](#対応マーケット)
- [クイックスタート](#クイックスタート)
- [使い方](#使い方)
- [MCPサーバー統合](#mcpサーバー統合)
- [プロジェクト構成](#プロジェクト構成)
- [API リファレンス](#api-リファレンス)
- [開発](#開発)
- [トラブルシューティング](#トラブルシューティング)
- [ロードマップ](#ロードマップ)
- [コントリビューション](#コントリビューション)
- [ライセンス](#ライセンス)

---

## 概要

**FX-Kline**は、外国為替市場のOHLC（始値・高値・安値・終値）データを効率的に取得・処理するPythonアプリケーションです。CLIユーティリティとModel Context Protocol (MCP)サーバーを提供し、手動分析とAI駆動の自動分析の両方に対応しています。※Streamlit UIは現在カルテ用途で利用できないため、CLI/MCPを利用してください。

### なぜFX-Klineを選ぶのか？

- 🚀 **並列データ取得**: 複数の通貨ペア・時間足を同時に取得（asyncio実装）
- 🌏 **完全な日本時間対応**: UTC→JST自動変換、DST（夏時間）対応
- 📊 **営業日フィルタリング**: FX市場の営業日に基づく高度なフィルタリング
- 🤖 **AI統合**: Claude Desktop/Cline対応のMCPサーバー
- 💾 **多様なエクスポート**: CSV、JSON、クリップボード対応
- 🎯 **使いやすさ**: WebUI、Python API、MCPの3つのインターフェース

### トレーディング日ポリシー（NYクローズ基準）

- FXの「1日」は NYクローズ（US/Eastern 17:00）= JST 6:00（夏時間）/7:00（冬時間）で切り替え。
- 日足（`interval="1d"`）での `period="Nd"` は「直近N本の確定済みトレーディング日」のみ。進行中のトレーディング日は除外。
- インターデイ（1m〜4h）での `period="Nd"` は「直近Nトレーディング日」扱い。N-1日はフル、最後の1日は進行中分の確定バーのみを含める。
- タイムスタンプはUTC/JSTをそのまま保持し、31:00のような仮想時刻にはしない。境界判定はロジック／表示側でラベル付け。
- ゴールド（`XAUUSD`）は Twelve Data (`XAU/USD`) から取得し、リクエストで `timezone=UTC` を指定したうえで UTC→JST に変換して保存します（yfinance/GC=F は精度課題のため不採用）。

---

## 主要機能

### 🔥 コア機能

| 機能 | 説明 |
|------|------|
| **並列取得** | asyncioによる高速な複数リクエスト処理 |
| **タイムゾーン変換** | UTC→JST自動変換、米国・欧州DSTサポート |
| **営業日フィルタ** | 市場別（FX/商品）の週末・休日除外 |
| **データ検証** | Pydanticによる型安全なデータモデル |
| **エラーハンドリング** | 詳細なエラー分類と復旧戦略 |

### 🎨 インターフェース

#### 1️⃣ CLIユーティリティ
- scripts/fetch_intraday_cli.py で手早く一括取得
- uv run python scripts/fetch_intraday_cli.py --help で使い方
- CSV出力をそのまま下流処理へ渡せる

#### 2️⃣ Python API
- プログラマティックなアクセス
- バッチ処理対応
- カスタムワークフロー統合

#### 3️⃣ MCP サーバー ✨
- Claude Desktopから自然言語で操作
- AI分析との自動連携
- 複数ツールとの連携

---

## デモ

### Python API

```python
from fx_kline.core import OHLCRequest, fetch_batch_ohlc_sync

# 複数リクエストを並列取得
requests = [
    OHLCRequest(pair="USDJPY", interval="1d", period="30d"),
    OHLCRequest(pair="EURUSD", interval="1h", period="5d"),
]

response = fetch_batch_ohlc_sync(requests)
print(f"取得完了: {response.total_succeeded}/{response.total_requested}")
```

### MCP サーバー

```
ユーザー: 「ドル円の過去30日の日足データを取得して、トレンドを分析して」

Claude: データを取得して分析します...
        [fetch_ohlc ツールを自動実行]
        分析結果: 上昇トレンド継続中、移動平均線は...
```

---

## 対応マーケット

### 📈 通貨ペア（全8ペア）

| カテゴリ | 通貨ペア | コード |
|---------|---------|--------|
| **メジャー** | ドル円 | `USDJPY` |
| | ユーロドル | `EURUSD` |
| | ポンドドル | `GBPUSD` |
| | オーストラリアドル/米ドル | `AUDUSD` |
| **クロス** | ユーロ円 | `EURJPY` |
| | ポンド円 | `GBPJPY` |
| | オーストラリアドル円 | `AUDJPY` |
| **商品** | 金（ゴールド） | `XAUUSD` |

> 📝 **注記**: ゴールド（`XAUUSD`）は Twelve Data (`XAU/USD`) を使用し、`timezone=UTC` で受けてからJSTに変換します。API呼び出しやMCPツールでは `XAUUSD` を指定してください（yfinance/GC=F は非採用）。

### ⏱️ 時間足

- `5m` - 5分足
- `15m` - 15分足
- `1h` - 1時間足
- `4h` - 4時間足
- `1d` - 日足

> 📝 今後のアップデートで1分足、週足、月足にも対応予定

---

## クイックスタート

### 前提条件

- **Python**: 3.10以上
- **パッケージマネージャー**: [uv](https://github.com/astral-sh/uv)（推奨）または pip

### インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/Mako3333/FX-Kline.git
cd FX-Kline

# 2. 依存関係をインストール
uv sync

# 3. (オプション) MCPサーバーを使う場合
uv sync --extra mcp
```

### 起動方法

#### CLI（推奨）

```bash
# 15分足、プリセット通貨ペア（デフォルト）を1営業日分取得
uv run python scripts/fetch_intraday_cli.py --intervals 15m --period 1d

# 5分足でペアを指定
uv run python scripts/fetch_intraday_cli.py --pairs USDJPY EURUSD GBPUSD XAUUSD --intervals 5m --period 1d
```

#### MCPサーバー

[docs/MCP_SETUP.md](./docs/MCP_SETUP.md) の詳細ガイドを参照してください。

---

## 使い方

詳しい手順は [docs/INTRADAY_FETCH.md](./docs/INTRADAY_FETCH.md) を参照してください。

### 🖥️ CLI

```bash
# デフォルト（プリセットペア、15m、1d、週末除外）
uv run python scripts/fetch_intraday_cli.py

# 5mで6ペア、週末も含めたい場合
uv run python scripts/fetch_intraday_cli.py --pairs USDJPY EURUSD GBPUSD AUDJPY AUDUSD XAUUSD --intervals 5m --period 1d --no-exclude-weekends
```

### 💻 Python API

```python
from fx_kline.core import OHLCRequest, fetch_batch_ohlc_sync

# リクエスト作成
requests = [
    OHLCRequest(pair="USDJPY", interval="1d", period="30d"),
    OHLCRequest(pair="EURUSD", interval="1h", period="5d"),
    OHLCRequest(pair="XAUUSD", interval="15m", period="1d"),
]

# バッチ取得
response = fetch_batch_ohlc_sync(requests, exclude_weekends=True)

# 結果処理
for ohlc in response.successful:
    print(f"{ohlc.pair}: {ohlc.data_count}件のデータ")
    # データはohlc.rowsに格納
```

詳細は [API リファレンス](#api-リファレンス) を参照してください。

---

## MCPサーバー統合

<div align="center">

### 🤖 Claude Desktopと統合してAI分析を自動化

</div>

FX-KlineはModel Context Protocol (MCP)サーバーとして動作し、Claude DesktopやClineなどのAIツールから直接利用できます。

### ✨ MCPサーバーの利点

| 利点 | 説明 |
|------|------|
| 🗣️ **自然言語インターフェース** | 「ドル円のデータを取得して」で完結 |
| 🔄 **自動ワークフロー** | データ取得→分析→レポート生成を自動化 |
| 🧩 **ツール連携** | 他のMCPツールと組み合わせて高度な分析 |
| ⚡ **効率化** | コード不要、対話的なデータ探索 |

### 🚀 MCPクイックスタート

```bash
# 1. MCP依存関係をインストール
uv sync --extra mcp

# 2. Claude Desktop設定ファイルを編集
# macOS/Linux: ~/.config/claude/claude_desktop_config.json
# Windows: %APPDATA%\Claude\claude_desktop_config.json
```

**設定例**:

```json
{
  "mcpServers": {
    "fx-kline": {
      "command": "uv",
      "args": ["run", "python", "/path/to/FX-Kline/run_mcp_server.py"]
    }
  }
}
```

> **⚠️ 重要**: `/path/to/FX-Kline/` の部分は、ご自身の環境における **絶対パス** に置き換えてください。
>
> 例: macOS/Linuxの場合 `/Users/username/projects/FX-Kline/run_mcp_server.py`
> 例: Windowsの場合 `C:\Users\username\projects\FX-Kline\run_mcp_server.py`

詳細なセットアップ手順とユースケースは **[docs/MCP_SETUP.md](./docs/MCP_SETUP.md)** を参照してください。

### 🛠️ 実装済みMCPツール

#### 推奨ツール（v0.2.0+）

| ツール名 | 機能 | 用途 |
|---------|------|------|
| `get_intraday_ohlc` | 日中足データ取得（1m-4h） | スキャルピング・デイトレード |
| `get_daily_ohlc` | 日足以上データ取得（1d-1mo） | スイング・ポジション取引 |
| `get_ohlc_batch` | 複数リクエストの並列バッチ取得 | 相関分析・ポートフォリオ分析 |
| `list_pairs` | 利用可能な通貨ペア一覧 | ペアの確認・検証 |
| `list_timeframes` | 利用可能な時間足一覧 | 時間足の確認・検証 |
| `ping` | サーバーヘルスチェック | 接続確認・機能検出 |

#### 非推奨ツール（2026-05-16削除予定）

| ツール名 | 移行先 |
|---------|--------|
| `fetch_ohlc` | `get_intraday_ohlc` / `get_daily_ohlc` |
| `fetch_ohlc_batch` | `get_ohlc_batch` |
| `list_available_pairs` | `list_pairs` |
| `list_available_timeframes` | `list_timeframes` |

> 📖 移行ガイドは **[docs/MIGRATION.md](./docs/MIGRATION.md)** を参照してください。

#### MCP 2025仕様対応 ✨

- **ツールアノテーション**: readOnlyHint, idempotentHint, openWorldHint対応
- **拡張エラーハンドリング**: category, hint, recoverable, suggested_toolsフィールド
- **Completions機能**: パラメータ自動補完（pair, interval, period）
- **改善された説明文**: 使用シーン、関連ツールのヒント付き

### 📚 使用例

```
ユーザー: 「USDJPY、EURUSD、GBPUSDの1時間足を過去5日分取得して、相関を分析して」

Claude: データを取得します...
        [get_ohlc_batch ツールを実行]

        相関分析結果:
        - USDJPY vs EURUSD: -0.65 (負の相関)
        - USDJPY vs GBPUSD: -0.72 (強い負の相関)
        - EURUSD vs GBPUSD: +0.89 (強い正の相関)

        分析: ドル円は他の通貨ペアと逆相関...
```

---

## プロジェクト構成

```
fx-kline/
├── src/fx_kline/
│   ├── core/                      # コアビジネスロジック
│   │   ├── models.py              # Pydanticデータモデル
│   │   ├── validators.py          # 入力検証・定数定義
│   │   ├── timezone_utils.py      # UTC↔JST変換、DST対応
│   │   ├── business_days.py       # 営業日フィルタリング
│   │   └── data_fetcher.py        # 並列データ取得
│   ├── mcp/                       # MCPサーバー実装
│   │   ├── server.py              # MCPサーバー本体
│   │   └── tools.py               # MCPツール定義
│   └── ui/
│       └── streamlit_app.py       # 旧Streamlit UI（現在は非推奨/停止中）
├── main.py                        # 旧Streamlit起動スクリプト（停止中）
├── run_mcp_server.py              # MCPサーバー起動スクリプト
├── claude_desktop_config.example.json  # Claude Desktop設定例
├── test_mcp_tools.py              # MCPツールテスト
├── pyproject.toml                 # プロジェクト設定
├── README.md                      # このファイル
├── AGENTS.md                      # 開発ガイドライン
└── docs/                          # ドキュメント
    ├── MCP_SETUP.md               # MCPセットアップガイド
    ├── MIGRATION.md                # 移行ガイド
    └── SOW.md                     # 仕様書
```

### アーキテクチャの特徴

- **3層構造**: Core（ビジネスロジック）、UI（プレゼンテーション）、MCP（統合層）
- **完全分離**: 各層が独立しており、相互に影響しない
- **再利用性**: CoreモジュールはUI/MCPどちらからも利用可能
- **拡張性**: 新しいインターフェースの追加が容易

---

## API リファレンス

### OHLCRequest

単一データ取得リクエストの定義。

```python
from fx_kline.core import OHLCRequest

request = OHLCRequest(
    pair="USDJPY",      # 通貨ペアコード
    interval="1h",      # 時間足（5m, 15m, 1h, 4h, 1d）
    period="30d"        # 取得期間（1d, 5d, 1mo, 3mo, 1y など）
)
```

### fetch_batch_ohlc_sync()

複数リクエストを並列取得。

```python
from fx_kline.core import fetch_batch_ohlc_sync

response = fetch_batch_ohlc_sync(
    requests=[...],           # OHLCRequestのリスト
    exclude_weekends=True     # 週末データを除外
)

# レスポンス
response.successful          # 成功したOHLCDataのリスト
response.failed             # 失敗したFetchErrorのリスト
response.total_requested    # リクエスト総数
response.total_succeeded    # 成功数
response.total_failed       # 失敗数
response.summary            # サマリー文字列
```

### エクスポート関数

```python
from fx_kline.core import export_to_csv, export_to_json

# CSV形式
csv_string = export_to_csv(ohlc_data)

# JSON形式
json_string = export_to_json(ohlc_data)
```

### タイムゾーンユーティリティ

```python
from fx_kline.core import (
    get_jst_now,                    # 現在のJST時刻
    utc_to_jst,                     # UTC→JST変換
    get_us_market_hours_in_jst,     # NY市場営業時間（JST）
    is_us_dst_active                # 米国DST判定
)
```

### 営業日ユーティリティ

```python
from fx_kline.core import (
    get_business_days_back,         # N営業日前の日付
    count_business_days,            # 期間内の営業日数
    filter_business_days            # DataFrameから週末を除外
)
```

詳細なAPIドキュメントは[こちら](./docs/API.md)（今後追加予定）

---

## 開発

### テストの実行

```bash
# 基本的なデータ取得テスト
uv run python test_fetch.py

# MCPツールのテスト
uv run python test_mcp_tools.py

# デバッグ（yfinanceデータ構造確認）
uv run python debug_fetch.py
```

### コード品質

```bash
# フォーマット
uv run black .

# リント
uv run ruff check --fix .
```

### 依存関係の管理

```bash
# 依存関係の更新
uv sync --upgrade

# 特定のグループのみインストール
uv sync --extra mcp        # MCP用
uv sync --extra dev        # 開発用
```

---

## トラブルシューティング

### よくある問題

<details>
<summary><b>データが取得できない</b></summary>

1. インターネット接続を確認
2. yfinanceのバージョン確認: `uv pip list | grep yfinance`
3. 通貨ペアコードが正しいか確認（大文字で指定）
4. デバッグスクリプトを実行: `uv run python debug_fetch.py`
</details>

<details>
<summary><b>MCPサーバーが認識されない</b></summary>

1. Claude Desktopを完全に終了
2. 設定ファイルのパスが絶対パスか確認
3. `uv sync --extra mcp` で依存関係をインストール
4. Claude Desktopを再起動
</details>

<details>
<summary><b>タイムゾーンが正しくない</b></summary>

- すべてのデータは自動的にJST（日本標準時）に変換されます
- `get_jst_now()` で現在時刻を確認できます
- DST（夏時間）も自動対応
</details>

### エラータイプ一覧

| エラー | 原因 | 対策 |
|-------|------|------|
| `NoDataAvailable` | 指定期間にデータなし | 取得期間を延長 |
| `AllWeekendData` | 取得期間が全て週末 | 取得期間を延長 |
| `ValidationError` | 不正な通貨ペア/時間足 | サポートされているペアを確認 |
| `TypeError` | データ型変換エラー | yfinanceのバージョン確認 |

---

## ロードマップ

### 🎯 現在の状態（v0.1.0）

- ✅ CLIユーティリティ
- ✅ Python API
- ✅ MCPサーバー統合
- ✅ 8通貨ペア対応
- ✅ 5種類の時間足

### 🚀 今後の予定

#### v0.2.0（次期バージョン）
- [ ] 1分足、週足、月足のサポート
- [ ] データキャッシング機能
- [ ] レート制限対策の強化
- [ ] 包括的なテストスイート

#### v0.3.0
- [ ] その他の通貨ペア追加
- [ ] カスタムインジケーター機能
- [ ] データベース保存オプション
- [ ] REST API提供

#### v1.0.0
- [ ] 完全な英語ドキュメント
- [ ] Docker対応
- [ ] Webホスティング対応
- [ ] プレミアムデータソース対応

ご意見・ご要望は[Issues](https://github.com/Mako3333/FX-Kline/issues)でお知らせください。

---

## コントリビューション

FX-Klineへの貢献を歓迎します！

### コントリビューション方法

1. **Issue作成**: バグ報告や機能リクエストは[Issues](https://github.com/Mako3333/FX-Kline/issues)へ
2. **プルリクエスト**:
   - フォークしてブランチを作成
   - コードを実装
   - テストを追加
   - PRを作成

### 開発ガイドライン

- Python 3.10+対応
- 4スペースインデント
- Type hintsを使用
- Pydanticモデルでデータ検証
- コミットメッセージは英語で（例: `feat: add new feature`）

詳細は[AGENTS.md](./AGENTS.md)を参照してください。

### コントリビューター

このプロジェクトに貢献してくださった方々に感謝します。

---

## ライセンス

このプロジェクトは[MITライセンス](./LICENSE)の下で公開されています。

```
MIT License

Copyright (c) 2025 Mako3333

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 謝辞

- [yfinance](https://github.com/ranaroussi/yfinance) - 市場データ提供
- [Streamlit](https://streamlit.io/) - 旧WebUIフレームワーク（現在停止中）
- [Model Context Protocol](https://modelcontextprotocol.io/) - AI統合標準
- [Anthropic Claude](https://www.anthropic.com/) - AI分析パートナー

---

## サポート・お問い合わせ

- 🐛 **バグ報告**: [GitHub Issues](https://github.com/Mako3333/FX-Kline/issues)
- 💡 **機能リクエスト**: [GitHub Issues](https://github.com/Mako3333/FX-Kline/issues)
- 📧 **その他**: GitHubのDiscussionsまたはIssuesで

---

<div align="center">

**[⬆ トップに戻る](#fx-kline)**

Made with ❤️ by [Mako3333](https://github.com/Mako3333)

⭐ このプロジェクトが役に立ったら、スターをお願いします！

</div>
