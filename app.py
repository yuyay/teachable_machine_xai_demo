import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from typing import Tuple
import zipfile
import os
import tempfile


class TeachableMachineModel:
    """Wrapper for Teachable Machine TensorFlow models"""
    
    def __init__(self, model_path: str, labels_path: str = None):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self._load_class_names(labels_path) if labels_path else None
        self._analyze_model_structure()
    
    def _load_class_names(self, labels_path: str) -> list:
        """Load class names from labels.txt file"""
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            class_names = []
            for line in lines:
                line = line.strip()
                if line:
                    # Format: "0 iPhone" -> extract "iPhone"
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        class_names.append(parts[1])
                    else:
                        class_names.append(parts[0])
            
            return class_names
        except Exception as e:
            print(f"Error loading labels: {e}")
            return None
    
    def _analyze_model_structure(self):
        """Analyze and print the Teachable Machine model structure"""
        print("=== Teachable Machine Model Analysis ===")
        print(f"Model type: {type(self.model)}")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print(f"Total parameters: {self.model.count_params():,}")
        
        print("\nModel Summary:")
        self.model.summary()
        
        print("\nDetailed Layer Structure:")
        for i, layer in enumerate(self.model.layers):
            print(f"Layer {i}: {layer.name}")
            print(f"  Type: {layer.__class__.__name__}")
            print(f"  Output shape: {layer.output_shape}")
            
            # If it's a Sequential/Functional model, show sublayers
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                print(f"  Sublayers ({len(layer.layers)}):")
                for j, sublayer in enumerate(layer.layers):
                    print(f"    {j}: {sublayer.name} ({sublayer.__class__.__name__}) - {sublayer.output_shape}")
                    
                    # Check for convolutional layers
                    if 'conv' in sublayer.__class__.__name__.lower():
                        print(f"      -> Convolutional layer found!")
            print()
        
        print("=" * 45)
        
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Predict class probabilities for an image using Teachable Machine preprocessing"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize and center crop image using PIL for consistency with Teachable Machine
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        # Use ImageOps.fit for center cropping (same as Teachable Machine)
        pil_image = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        image_array = np.asarray(pil_image)
        
        # Apply Teachable Machine normalization: (pixel / 127.5) - 1
        normalized_image = (image_array.astype(np.float32) / 127.5) - 1
        
        # Add batch dimension
        image_batch = np.expand_dims(normalized_image, axis=0)
        
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        return predictions[0], predicted_class

def extract_teachable_machine_files(zip_file) -> Tuple[str, str]:
    """Extract keras_model.h5 and labels.txt from uploaded zip file"""
    import shutil
    
    # Create temp directory in current working directory for Streamlit Cloud compatibility
    temp_dir = "temp_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save uploaded file to temporary location
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        
        # Extract zip file
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find keras_model.h5 and labels.txt
        model_path = None
        labels_path = None
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file == "keras_model.h5":
                    # Copy to current directory
                    model_path = "temp_keras_model.h5"
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    shutil.copy(os.path.join(root, file), model_path)
                elif file == "labels.txt":
                    # Copy to current directory
                    labels_path = "temp_labels.txt"
                    if os.path.exists(labels_path):
                        os.remove(labels_path)
                    shutil.copy(os.path.join(root, file), labels_path)
        
        if not model_path:
            raise FileNotFoundError("keras_model.h5 not found in zip file")
        
        return model_path, labels_path
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors

class TensorFlowXAIVisualizer:
    """TensorFlow/Keras implementation of XAI visualization using Integrated Gradients"""
    
    def __init__(self, model, layer_name: str = None):
        self.model = model
        self.layer_name = layer_name
        self.target_conv_layer = None
        self._find_target_conv_layer()
        
    def _find_target_conv_layer(self):
        """Find the best convolutional layer for GradCAM visualization"""
        conv_layers = []
        
        print("=== Finding Target Conv Layer ===")
        
        # Search deeper into nested models (like MobileNet in Teachable Machine)
        def search_layers_recursively(layers, path=""):
            for i, layer in enumerate(layers):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                layer_type = layer.__class__.__name__.lower()
                
                # Check if this layer itself is a conv layer
                if any(keyword in layer_type for keyword in ['conv', 'separable', 'depthwise']):
                    conv_layers.append((layer, current_path, layer.name))
                    print(f"Found conv layer: {layer.name} at {current_path}")
                
                # Recursively search in sublayers
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    search_layers_recursively(layer.layers, current_path)
        
        # Start recursive search
        search_layers_recursively(self.model.layers)
        
        if conv_layers:
            # Use the last convolutional layer found
            self.target_conv_layer = conv_layers[-1]
            print(f"Using target layer: {self.target_conv_layer[2]} at {self.target_conv_layer[1]}")
        else:
            # Fallback: try to use the feature extractor output
            print("No conv layers found, trying feature extractor...")
            # For Teachable Machine, use the output of the first sequential (feature extractor)
            if len(self.model.layers) > 0 and hasattr(self.model.layers[0], 'layers'):
                feature_extractor = self.model.layers[0].layers[0]  # model1
                self.target_conv_layer = (feature_extractor, "[0][0]", "feature_extractor")
                print(f"Using feature extractor: {feature_extractor.name}")
            else:
                self.target_conv_layer = None
                print("No suitable layers found!")
        
        print("=" * 35)
    
    def generate_explanation(self, image: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate XAI explanation visualization"""
        return self._generate_explanation(image, class_idx)
    
    def _generate_explanation(self, image: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate XAI explanation using Integrated Gradients for complex models"""
        img_array = self._preprocess_image(image)
        
        # Try to use a simpler approach for nested/transfer learning models
        try:
            return self._generate_integrated_gradients(img_array, image, class_idx)
        except Exception as e:
            print(f"Integrated gradients failed: {e}")
            # Fallback to guided backpropagation
            return self._generate_guided_backprop(img_array, image, class_idx)
    
    def _generate_integrated_gradients(self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate visualization using Integrated Gradients method"""
        baseline = tf.zeros_like(img_array)
        m_steps = 50
        
        # Generate path from baseline to input
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
        
        def interpolate_images(baseline, image, alphas):
            alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
            baseline_x = tf.expand_dims(baseline, axis=0)
            input_x = tf.expand_dims(image, axis=0)
            delta = input_x - baseline_x
            images = baseline_x + alphas_x * delta
            return images
        
        # Generate interpolated inputs
        interpolated_images = interpolate_images(baseline[0], img_array[0], alphas)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            predictions = self.model(interpolated_images)
            class_outputs = predictions[:, class_idx]
        
        grads = tape.gradient(class_outputs, interpolated_images)
        
        # Average gradients and compute integrated gradients
        avg_grads = tf.reduce_mean(grads, axis=0)
        integrated_grads = (img_array[0] - baseline[0]) * avg_grads
        
        # Convert to importance map
        importance = tf.reduce_mean(tf.abs(integrated_grads), axis=-1)
        
        # Normalize
        importance = (importance - tf.reduce_min(importance)) / (tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8)
        heatmap = importance.numpy()
        
        # Apply smoothing
        heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
        
        return self._create_visualization(original_image, heatmap)
    
    def _generate_guided_backprop(self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate visualization using guided backpropagation"""
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = self.model(img_array)
            class_output = predictions[0][class_idx]
        
        # Compute gradients with respect to input
        grads = tape.gradient(class_output, img_array)
        
        if grads is None:
            raise ValueError("Could not compute gradients")
        
        # Convert gradients to importance map
        grads = tf.abs(grads[0])  # Take absolute value and remove batch dimension
        importance = tf.reduce_mean(grads, axis=-1)  # Average over color channels
        
        # Normalize
        importance = (importance - tf.reduce_min(importance)) / (tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8)
        heatmap = importance.numpy()
        
        # Apply smoothing
        heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
        
        return self._create_visualization(original_image, heatmap)
    
    def _smooth_heatmap(self, heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to heatmap for better visualization"""
        try:
            from scipy.ndimage import gaussian_filter
            # Apply Gaussian filter for smoothing
            smoothed = gaussian_filter(heatmap, sigma=sigma)
            return smoothed
        except ImportError:
            # Fallback: use OpenCV Gaussian blur if scipy is not available
            kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))  # Ensure odd kernel size >= 3
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)
            return smoothed
    
    def _create_visualization(self, image: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization using 224x224 center-cropped image and heatmap"""
        # Convert original image to RGB
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply the same center cropping as used for model input
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_image_cropped = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
        cropped_image = np.asarray(pil_image_cropped)
        
        # Use the center-cropped 224x224 image
        rgb_img = cropped_image.astype(np.float32) / 255.0
        
        # Create smooth heatmap overlay with better color mapping
        # Apply smoothing to the heatmap (already 224x224)
        kernel_size = 11  # Fixed kernel size for 224x224
        heatmap_smooth = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 2.0)
        
        # Enhance contrast for better visibility
        heatmap_enhanced = np.power(heatmap_smooth, 0.7)  # Gamma correction for better contrast
        heatmap_enhanced = np.clip(heatmap_enhanced, 0, 1)
        
        # Use JET colormap consistently (red = high importance, blue = low importance)
        heatmap_colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_enhanced), cv2.COLORMAP_JET)
        # Convert BGR to RGB for proper overlay
        heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Overlay the heatmap on the original image (both in RGB format)
        # Use alpha blending based on heatmap intensity for better visibility
        alpha = heatmap_enhanced[..., np.newaxis] * 0.6 + 0.2  # Dynamic alpha based on importance
        overlay = (1 - alpha) * rgb_img + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        # Convert back to BGR for consistency with OpenCV
        overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Create heatmap for separate display (using the original 224x224 heatmap)
        heatmap_display = np.uint8(255 * heatmap)
        
        return overlay_bgr, heatmap_display
    
    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """Preprocess image using Teachable Machine format"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize and center crop using PIL for consistency with Teachable Machine
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        # Use ImageOps.fit for center cropping (same as Teachable Machine)
        pil_image = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
        
        # Convert back to numpy array
        image_array = np.asarray(pil_image)
        
        # Apply Teachable Machine normalization: (pixel / 127.5) - 1
        normalized_image = (image_array.astype(np.float32) / 127.5) - 1
        
        # Add batch dimension
        img_array = np.expand_dims(normalized_image, axis=0)
        
        return tf.convert_to_tensor(img_array)

def main():
    # Set page config for better mobile compatibility
    st.set_page_config(
        page_title="XAI Demo - Teachable Machine + GradCAM", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 XAI Demo: Teachable Machine + Integrated Gradients")
    st.markdown("Teachable Machineで学習したモデルを使用してWebカメラ画像を分類し、Integrated Gradientsで重要領域を可視化します。")
    
    # Privacy notice
    with st.expander("🔒 プライバシーについて"):
        st.markdown("""
        **このアプリのプライバシーポリシー:**
        - アップロードされた画像はサーバーに保存されません
        - モデルファイルは一時的にのみ処理され、セッション終了後に削除されます
        - 撮影された画像は分析後にメモリから削除されます
        - 個人情報は収集されません
        """)
    
    # サイドバーでモデル設定
    st.sidebar.header("モデル設定")
    
    # ZIPファイルのアップロード
    uploaded_zip = st.sidebar.file_uploader(
        "Teachable Machineモデル(.zip)をアップロード",
        type=['zip'],
        help="Google Teachable Machineでエクスポートしたzipファイルをアップロードしてください（keras_model.h5とlabels.txtを含む）"
    )
    
    if uploaded_zip is not None:
        try:
            # ZIPファイルから必要なファイルを抽出
            with st.spinner("モデルファイルを抽出中..."):
                model_path, labels_path = extract_teachable_machine_files(uploaded_zip)
            
            # モデルとXAI可視化の初期化
            tm_model = TeachableMachineModel(model_path, labels_path)
            xai_visualizer = TensorFlowXAIVisualizer(tm_model.model)
            
            st.success("モデルが正常に読み込まれました！")
            
            # Display model info
            if tm_model.class_names:
                st.info(f"クラス数: {len(tm_model.class_names)} クラス")
                with st.expander("クラス一覧"):
                    for i, name in enumerate(tm_model.class_names):
                        st.write(f"{i}: {name}")
            
            # メインコンテンツ
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("📷 Webカメラ")
                
                # Webカメラからの画像取得
                camera_input = st.camera_input("写真を撮影してください")
                
                if camera_input is not None:
                    # 画像の読み込みと変換
                    image = Image.open(camera_input)
                    image_array = np.array(image)
                    
                    # RGB to BGR (OpenCV用)
                    if len(image_array.shape) == 3:
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image_array
                    
                    # 予測実行
                    predictions, predicted_class = tm_model.predict(image_bgr)
                    
                    # 結果表示
                    st.subheader("🎯 予測結果")
                    
                    if tm_model.class_names and len(tm_model.class_names) > predicted_class:
                        st.write(f"**予測クラス:** {tm_model.class_names[predicted_class]}")
                    else:
                        st.write(f"**予測クラス:** Class {predicted_class}")
                    
                    st.write(f"**信頼度:** {predictions[predicted_class]:.2%}")
                    
                    # 全クラスの確率表示
                    st.subheader("📊 全クラスの確率")
                    for i, prob in enumerate(predictions):
                        label = tm_model.class_names[i] if tm_model.class_names and i < len(tm_model.class_names) else f"Class {i}"
                        st.write(f"{label}: {prob:.2%}")
            
            with col2:
                st.header("🔍 XAI可視化")
                
                if camera_input is not None:
                    # XAI可視化の生成
                    with st.spinner("Integrated Gradientsを生成中..."):
                        try:
                            overlay, heatmap = xai_visualizer.generate_explanation(image_bgr, predicted_class)
                            
                            # BGR to RGB (表示用)
                            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            
                            st.image(overlay_rgb, caption="Integrated Gradients可視化結果", use_column_width=True)
                            
                            # ヒートマップのみも表示
                            st.subheader("🌡️ 重要度マップ")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(heatmap, cmap='jet')
                            ax.axis('off')
                            ax.set_title('Integrated Gradients Importance Map')
                            
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            st.error(f"XAI可視化生成エラー: {str(e)}")
                            st.info("モデルの予測処理で問題が発生した可能性があります。")
                        
                        # 説明文
                        st.info("**色の意味:** 赤い領域ほどモデルが分類の判断に重要視している部分、青い領域は重要度が低い部分です。")
                
                else:
                    st.info("Webカメラで写真を撮影すると、XAI可視化結果が表示されます。")
        
        except Exception as e:
            st.error(f"モデルの読み込みでエラーが発生しました: {str(e)}")
            st.info("Teachable Machineでエクスポートしたzipファイルを使用してください。keras_model.h5とlabels.txtが含まれている必要があります。")
        
        finally:
            # Clean up temporary files for security
            for temp_file in ["temp_keras_model.h5", "temp_labels.txt"]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    else:
        st.info("👈 サイドバーからTeachable Machineモデル（zip）をアップロードしてください。")
        
        # 使用方法の説明
        st.markdown("""
        ## 📝 使用方法
        
        1. **モデルの準備**
           - [Google Teachable Machine](https://teachablemachine.withgoogle.com/)で画像分類モデルを作成
           - モデルをTensorFlow形式でエクスポート（zipファイルをダウンロード）
           
        2. **アプリの設定**
           - サイドバーからzipファイルをアップロード
           - keras_model.h5とlabels.txtが自動で読み込まれます
           
        3. **画像分類とXAI**
           - Webカメラで写真を撮影
           - 自動で分類結果とIntegrated Gradients可視化が表示されます
        
        ## 🎯 XAI（説明可能AI）について
        
        このデモでは**Integrated Gradients**を使用して、AIがどの部分を重要視して判断しているかを可視化しています。
        - Teachable MachineのTensorFlowモデルを直接使用
        - 転移学習モデルに最適化された可視化手法
        - 赤い領域ほど、モデルが分類の決定に重要と判断した部分です。
        """)

if __name__ == "__main__":
    main()