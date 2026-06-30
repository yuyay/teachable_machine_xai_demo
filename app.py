"""Streamlit XAI demo: Teachable Machine + Integrated Gradients.

UI layer only. Model loading lives in model.py, XAI in xai.py, constants in
config.py. Designed for Cloud Run with low per-instance concurrency.
"""
import threading

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

import config
from model import load_model_cached, hash_bytes, TeachableMachineModel
from xai import TensorFlowXAIVisualizer

# In-instance guard so concurrent sessions don't all spike memory at once.
_XAI_SEMAPHORE = threading.BoundedSemaphore(config.XAI_SEMAPHORE)


def configure_tensorflow() -> None:
    """Configure TF CPU thread limits for this session.

    Guarded once per Streamlit session via session_state; if the TF runtime
    is already initialized in this process, the RuntimeError is ignored
    (thread settings are process-global and only appltable before init).
    """
    if st.session_state.get("_tf_configured"):
        return
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_THREADS)
    except RuntimeError:
        # TF runtime already initialized in this process; safe to ignore.
        pass
    st.session_state["_tf_configured"] = True


def _render_predictions(
    tm_model: TeachableMachineModel,
    predictions: np.ndarray,
    predicted_class: int,
) -> None:
    """Render the top prediction and per-class probabilities to the UI."""
    st.subheader("🎯 予測結果")
    if tm_model.class_names and len(tm_model.class_names) > predicted_class:
        st.write(f"**予測クラス:** {tm_model.class_names[predicted_class]}")
    else:
        st.write(f"**予測クラス:** Class {predicted_class}")
    st.write(f"**信頼度:** {predictions[predicted_class]:.2%}")
    st.subheader("📊 全クラスの確率")
    for i, prob in enumerate(predictions):
        if tm_model.class_names and i < len(tm_model.class_names):
            label = tm_model.class_names[i]
        else:
            label = f"Class {i}"
        st.write(f"{label}: {prob:.2%}")


def _render_xai(
    xai_visualizer: TensorFlowXAIVisualizer,
    image_bgr: np.ndarray,
    predicted_class: int,
) -> None:
    """Run the (semaphore-guarded) XAI computation and render the overlay."""
    with st.spinner("Integrated Gradientsを生成中..."):
        try:
            with _XAI_SEMAPHORE:
                overlay, heatmap = xai_visualizer.generate_explanation(
                    image_bgr, predicted_class
                )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption="Integrated Gradients可視化結果",
                use_container_width=True,
            )
            st.subheader("🌡️ 重要度マップ")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Integrated Gradients Importance Map")
            st.pyplot(fig)
            plt.close()
        except Exception as e:  # noqa: BLE001 - surface to UI
            st.error(f"XAI可視化生成エラー: {str(e)}")
            st.info("モデルの予測処理で問題が発生した可能性があります。")
        st.info(
            "**色の意味:** 赤い領域ほどモデルが分類の判断に重要視している部分、"
            "青い領域は重要度が低い部分です。"
        )


def _render_instructions() -> None:
    """Render the landing instructions shown before a model is uploaded."""
    st.info("👈 サイドバーからTeachable Machineモデル（zip）をアップロードしてください。")
    st.markdown(
        """
        ## 📝 使用方法

        1. **モデルの準備** — [Teachable Machine](https://teachablemachine.withgoogle.com/)
           で画像分類モデルを作成し、TensorFlow 形式（zip）でエクスポート
        2. **アプリの設定** — サイドバーから zip をアップロード
           （keras_model.h5 と labels.txt が自動で読み込まれます）
        3. **画像分類と XAI** — Webカメラで撮影すると、分類結果と
           Integrated Gradients 可視化が表示されます
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="XAI Demo - Teachable Machine + Integrated Gradients",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_tensorflow()

    st.title("🤖 XAI Demo: Teachable Machine + Integrated Gradients")
    st.markdown(
        "Teachable Machine で学習したモデルを使用して Web カメラ画像を分類し、"
        "Integrated Gradients で重要領域を可視化します。"
    )

    with st.expander("🔒 プライバシーについて"):
        st.markdown(
            """
            - アップロードされた画像はサーバーに保存されません
            - モデルファイルは一時的にのみ処理され、推論後にメモリ上でのみ保持されます
            - 撮影された画像は分析後にメモリから削除されます
            - 個人情報は収集されません
            """
        )

    st.sidebar.header("モデル設定")
    uploaded_zip = st.sidebar.file_uploader(
        "Teachable Machineモデル(.zip)をアップロード",
        type=["zip"],
        help="keras_model.h5 と labels.txt を含む zip をアップロードしてください",
    )

    if uploaded_zip is None:
        _render_instructions()
        return

    try:
        zip_bytes = uploaded_zip.getvalue()
        with st.spinner("モデルを読み込み中..."):
            tm_model = load_model_cached(hash_bytes(zip_bytes), zip_bytes)
        xai_visualizer = TensorFlowXAIVisualizer(tm_model.model)
        st.success("モデルが正常に読み込まれました！")
    except Exception as e:  # noqa: BLE001 - surface to UI
        st.error(f"モデルの読み込みでエラーが発生しました: {str(e)}")
        st.info(
            "Teachable Machine でエクスポートした zip（keras_model.h5 と "
            "labels.txt を含む）を使用してください。"
        )
        return

    if tm_model.class_names:
        st.info(f"クラス数: {len(tm_model.class_names)} クラス")
        with st.expander("クラス一覧"):
            for i, name in enumerate(tm_model.class_names):
                st.write(f"{i}: {name}")

    col1, col2 = st.columns(2)
    with col1:
        st.header("📷 Webカメラ")
        camera_input = st.camera_input("写真を撮影してください")
        image_bgr: np.ndarray | None = None
        predicted_class: int = 0
        if camera_input is not None:
            image = Image.open(camera_input)
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            predictions, predicted_class = tm_model.predict(image_bgr)
            _render_predictions(tm_model, predictions, predicted_class)

    with col2:
        st.header("🔍 XAI可視化")
        if camera_input is not None and image_bgr is not None:
            _render_xai(xai_visualizer, image_bgr, predicted_class)
        else:
            st.info("Webカメラで写真を撮影すると、XAI可視化結果が表示されます。")


if __name__ == "__main__":
    main()
