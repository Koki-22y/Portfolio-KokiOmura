# Portfolio-KokiOmura
概要

このプロジェクトは、ひらがな画像を分類するための 畳み込みニューラルネットワーク（CNN） モデルを TensorFlow/Keras を用いて構築したものです。
ひらがな画像のデータセットを読み込み、学習・評価・推論まで行う一連のコードが含まれています。

主な機能：

画像の前処理（リサイズ・正規化）

データ拡張（Data Augmentation）

CNN モデルの構築・学習

学習曲線の可視化

混同行列の作成

任意の画像からひらがなを推論

📁 プロジェクト構成
project/
│
├── hiragana_images/        # トレーニング画像フォルダ
├── hiragana_model.keras    # 学習済みモデル（保存後に生成）
├── train.py                # モデル学習コード
├── predict.py              # モデル推論コード
└── README.md               # この説明書

使用技術

Python 3.x

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Seaborn

Scikit-learn

　1. データセットについて

フォルダ：hiragana_images/

各画像はファイル名からラベルを抽出
例：

a01.png → ラベル: a
ka12.png → ラベル: ka


画像はすべて 32×32 にリサイズし、ピクセル値を 0〜1 の範囲に正規化します。

データ拡張として、左右反転や回転、ズームなどを実施。

　 2. モデルの学習
実行方法
python train.py

使用される CNN モデル

Conv2D → MaxPooling （2セット）

Flatten

Dense(128)

Dense（クラス数）

データ拡張ありで訓練します。

学習後：

hiragana_model.keras としてモデルが保存されます

学習曲線（Loss）がポップアップ表示されます

混同行列が表示されます

　 3. 推論（予測）

学習済みモデルを使って、任意の画像のひらがなを予測できます。

実行方法
python predict.py


予測コードでは以下を実行します：

モデルを読み込み

テスト用フォルダから画像をランダムに取得

画像を 32×32 に加工

推論結果を出力

　4. ラベルマップ

ひらがなとモデル内のラベル番号は以下のように対応しています：

hiragana_list = ["kanaA", "kanaE", "kanaI", "kanaO", "kanaU"]
label_map = {i: hiragana_list[i] for i in range(len(hiragana_list))}

　5. 出力例
Predicted Hiragana for test_12.png: kanaA
Predicted Hiragana for test_77.png: kanaU
Predicted Hiragana for test_03.png: kanaI

　6. 必要ライブラリ
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn

 注意点

モデルのファイル形式は hiragana_model.keras として保存されますが、推論コード内では .h5 を読み込むようになっています。
→ 必要に応じて名称を合わせてください。

例：

model = load_model('hiragana_model.keras')


