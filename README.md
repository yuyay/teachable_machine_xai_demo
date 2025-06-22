# XAI Demo - Teachable Machine + Integrated Gradients

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Google Teachable Machineで学習した画像認識モデルを読み込み、Webカメラを使って画像を取得してIntegrated Gradientsで分類結果の重要領域をハイライトするStreamlitデモアプリケーションです。

## ✨ 機能

- **🎯 Teachable Machineモデル対応**: Google Teachable Machineでエクスポートしたzipファイルを直接読み込み
- **📷 リアルタイム画像分類**: Webカメラで撮影した画像をその場で分類
- **🔍 XAI可視化**: Integrated Gradientsを使用してモデルの判断根拠を可視化
- **🎨 スムースなヒートマップ**: 高品質なinterpolationによる美しい可視化
- **💻 直感的なUI**: Streamlitによる使いやすいWebインターフェース
- **🌐 クラウド対応**: Streamlit Cloudでの実行をサポート

## セットアップ

### 必要な環境

- Python 3.8以上
- Webカメラ（ブラウザからアクセス可能）

### インストール

1. リポジトリをクローンまたはダウンロード
```bash
git clone <repository-url>
cd xai_demo
```

2. 依存関係をインストール
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. モデルの準備

1. [Google Teachable Machine](https://teachablemachine.withgoogle.com/)にアクセス
2. 「Image Project」を選択
3. 画像データをアップロードして分類モデルを作成
4. 「Export Model」から「TensorFlow」→「Keras」を選択
5. 「Download my model」でzipファイルをダウンロード（`keras_model.h5`と`labels.txt`を含む）

### 2. アプリの起動

```bash
streamlit run app.py
```

ブラウザが自動で開き、アプリが表示されます。

### 3. デモの実行

1. **モデルアップロード**: サイドバーからzipファイルをアップロード
2. **画像撮影**: Webカメラで写真を撮影
3. **結果確認**: 分類結果とIntegrated Gradients可視化を確認

### 4. オンラインで試す

Streamlit Cloudでホストされているデモをこちらで試すことができます：
[Demo App](https://your-app-url.streamlit.app)

> **注意**: 初回アクセス時は、アプリの起動に1-2分程度かかる場合があります。

## 画面説明

### 左側パネル
- **予測結果**: 最も確率が高いクラスと信頼度
- **全クラス確率**: 各クラスの予測確率

### 右側パネル
- **XAI可視化**: 元画像にヒートマップを重ね合わせた結果
- **重要度マップ**: 重要度の可視化（赤い領域ほど重要）

## 技術仕様

- **フレームワーク**: Streamlit
- **機械学習**: TensorFlow/Keras
- **画像処理**: OpenCV, PIL
- **可視化**: Matplotlib
- **XAI手法**: Integrated Gradients

## Integrated Gradientsについて

Integrated Gradientsは、ディープラーニングモデルの判断根拠を可視化するXAI（説明可能AI）手法です。入力画像からベースライン（通常は黒い画像）への経路を統合することで、モデルが画像のどの部分に注目して分類判断を行っているかをヒートマップで表示します。

- **赤い領域**: モデルが重要視している部分
- **青い領域**: 判断にあまり寄与していない部分

## トラブルシューティング

### よくある問題

1. **モデル読み込みエラー**
   - Teachable MachineからTensorFlow/Keras形式でエクスポートしたzipファイルを使用してください
   - zipファイルに`keras_model.h5`と`labels.txt`が含まれていることを確認してください
   
2. **Webカメラアクセスエラー**
   - ブラウザでカメラのアクセス許可を有効にしてください
   - HTTPSまたはlocalhostでアクセスしてください

3. **XAI可視化生成エラー**
   - モデルの予測処理で問題が発生していないか確認してください

## 🤝 コントリビューション

バグ報告、機能要望、プルリクエストを歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 🙏 謝辞

- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) - 簡単にモデルを作成できる素晴らしいツール
- [Streamlit](https://streamlit.io/) - 美しいWebアプリを簡単に作成できるフレームワーク
- [Integrated Gradients論文](https://arxiv.org/abs/1703.01365) - XAI可視化の基礎となる研究

## 📞 サポート

質問や問題がある場合は、[Issues](https://github.com/your-username/xai_demo/issues)で報告してください。