# FX Technical Summary Pipeline

このリポジトリは、FX/CFD の OHLC データを日次で取得し、テクニカル分析サマリ（JSON）を自動生成するためのパイプラインです。  
GitHub Actions を利用して、毎日最新の市場状況を `data/` ディレクトリに蓄積します。

## 主な機能

- **データ取得**: `yfinance` および `Twelve Data` から OHLC データを取得
- **テクニカル分析**: SMA, EMA, RSI, ATR, サポート/レジスタンスラインの自動計算
- **自動保存**: GitHub Actions で集計結果を日付フォルダ（`data/YYYY/MM/DD/`）に自動コミット
- **マルチタイムフレーム**: 1h, 4h, 1d の複数時間足を統合したサマリ生成

## ディレクトリ構造

- `src/fx_kline/core`: データ取得・分析のコアロジック
- `scripts/`: パイプライン実行用スクリプト
- `data/`: 蓄積された日次サマリデータ
- `.github/workflows/`: GitHub Actions の設定

## クイックスタート

### 1. リポジトリの準備
1. このリポジトリを **Fork** します。
2. GitHub Secrets に `TWELVEDATA_API_KEY` を設定します（Gold などの銘柄を取得する場合のみ）。

### 2. GitHub Actions の有効化
`Actions` タブから **Daily OHLC Analysis** を選択し、`Run workflow` をクリックして手動実行するか、設定されたスケジュール（平日 07:05 JST）を待ちます。

### 3. ローカルでの実行
```bash
pip install -r requirements.txt

# 1. データの取得
python scripts/fetch_daily_data_local.py

# 2. 個別分析の生成
python ohlc_aggregator.py --input-dir ./csv_data --output-dir ./reports

# 3. サマリの統合
python consolidate_summaries.py --reports-dir ./reports --output-dir ./summary_reports

# 4. 日付フォルダへの整理
python scripts/prepare_daily_data.py
```

## データフォーマット

生成されるサマリ（`data/YYYY/MM/DD/summaries/{PAIR}_summary.json`）には、以下の情報が含まれます。

- `trend`: 相場の方向性（UP / DOWN / SIDEWAYS）
- `support_levels` / `resistance_levels`: 直近の主要な水平線（ATRを考慮した高値・安値）
- `rsi` (14): 売られすぎ・買われすぎの指標
- `atr` (14): 直近のボラティリティ（値幅）
- `sma`: 5, 13, 21期間の単純移動平均
    - `slope`: 傾き（up, flat, down など）
    - `ordering`: パーフェクトオーダーの判定（bullish, bearish, mixed）
- `ema`: 25, 75, 90, 200期間の指数平滑移動平均
    - `reaction`: 直近の反発判定（support_bounce, resistance_reject など）
- `time_of_day` (1hのみ): 時間帯ごとのリバーサル（反転）発生スコア

## ライセンス
MIT License
