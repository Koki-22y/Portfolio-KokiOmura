  概要

このリポジトリは、Feedzai が公開している TimeSHAP（Apache 2.0 License）のコードを基盤に、
時系列予測モデルにおける Explainability（説明可能性）の可視化方法を改良したプロジェクトです。

本研究（卒業論文）では、LSTM/GRU などの時系列予測モデルが
「どの時点の入力が予測にどれだけ影響したか」 をより直感的に理解できるよう、
TimeSHAP の可視化部分および WindowSHAP の処理を改変し、新しい視覚化手法を提案しました。

  目的

TimeSHAP の既存の可視化は：

重要度スコアが連続的に見えない

長い時系列の場合、どのタイムステップが主要因か読み取りづらい

Window SHAP の影響範囲が把握しづらい

といった課題がありました。

そこで本プロジェクトでは、

 1. TimeSHAP の出力形式を解析しやすく整理
 2. 特に WindowSHAP の結果を、時間方向に直感的に可視化
 3. 既存の散布図を改良し、ヒートマップ／強調表示に変換
 4. 予測の上昇・下降に寄与するインスタンスを色分けして表示

といった手法を実装しています。

  改良内容（あなたが実際に行ったこと）
 1. 既存 TimeSHAP コードの解読と再構築

モジュール全体を読み解き、依存関係を再整理。

SHAP 近似ロジックを理解したうえで、可視化部分の構造を変更。

 2. WindowSHAP のインパクトスコアを時系列軸で並べ替え

これにより、
「どの時間窓が予測に最も影響したか」
が明確に理解できるようにした。

 3. 新しい可視化メソッドを実装（Heatmap・Line Emphasis）

例：

タイムステップ × 特徴量の 2D ヒートマップ

重要なステップのみ濃く表示する「強調ラインプロット」

予測上昇に寄与 → 赤

予測下降に寄与 → 青
など、直感的な色分けを追加。

 4. Time Series Forecasting モデルに最適化した処理

特に LSTM の入力形式 (batch, timesteps, features) に合わせて
可視化スコアの reshape / aggregation を行った。

 5. 論文用に TimeSHAP 全体を Jupyter Notebook 化

学習 → 推論 → TimeSHAP 実行 → 可視化
を 1 ノートブック内で行う構成にした。

  リポジトリ構成（例）
timeshap-visualisation-enhanced/
│
├── src/
│   ├── timeshap_modified.py          # 改良した TimeSHAP コアロジック
│   ├── windowshap_modified.py        # WindowSHAP の修正版
│   └── viz_utils.py                  # 新しい可視化関数
│
├── notebooks/
│   ├── TimeSHAP_analysis.ipynb       # 学習→説明性分析のフルパイプライン
│   └── Visualisation_Demo.ipynb      # 提案手法の可視化デモ
│
├── models/
│   └── trained_model.pth or .keras   # 使用した LSTM/GRU モデル
│
├── data/
│   └── sample_timeseries.csv         # デモ用時系列データ
│
└── README.md

  セットアップ
1. 依存ライブラリをインストール
pip install -r requirements.txt


主な依存：

pytorch or tensorflow

numpy

pandas

matplotlib

seaborn

shap

timeshap (元リポジトリ)

  使い方（例）
① モデルに対する TimeSHAP 実行
from timeshap_modified import TimeShapExplainer

explainer = TimeShapExplainer(model, background_data)
scores = explainer.explain(input_sequence)

② WindowSHAP の結果表示
from windowshap_modified import plot_window_impact

plot_window_impact(scores)

③ 新しい Heatmap 可視化
from viz_utils import plot_timeshap_heatmap

plot_timeshap_heatmap(scores, feature_names)

  可視化例（概要）

（※ README 用なので文章で紹介）

時系列 × 特徴量のヒートマップ
→ どの時点・特徴が結果に影響したかが一目でわかる

重要タイムステップの強調ライン
→ 予測に影響を与えた部分だけ色濃く表示

WindowSHAP 影響範囲のバンド表示
→ SHAP がどのウィンドウを参照したかを視覚化

  ライセンスについて（重要）

本プロジェクトは以下に基づいています：

TimeSHAP は Apache License 2.0 のもとで公開されています。
商用利用が必要な場合は Feedzai へ問い合わせる必要があります。

あなたのリポジトリでは改変部分を明記する必要があります：

This repository includes modified components originally from
TimeSHAP (https://github.com/feedzai/timeshap), licensed under Apache 2.0.
Modifications by Koki Omura, 2024.


これは Apache 2.0 の要件を満たす適正な記述 です。

  関連研究（あなたの論文）

Thesis: Visualizing Explainability Tools for Time-Series Forecasting Models

University of Twente, 2024

Author: Koki Omura

内容：

TimeSHAP を改善した新しい可視化手法を提案

実験を通して、既存手法より解釈性が向上することを検証