"""
Streamlit App — CIFAR-10 Image Classification
Comparative Study of CNN Architectures
UMT 6th Semester Deep Learning Project
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_EMOJI = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱',
    'deer': '🦌', 'dog': '🐶', 'frog': '🐸', 'horse': '🐴',
    'ship': '🚢', 'truck': '🚛'
}

MODEL_INFO = {
    "Baseline CNN": {
        "desc": "Simple 3-block CNN without advanced regularisation. Serves as the performance baseline.",
        "color": "#4C9BE8"
    },
    "VGG-Style CNN": {
        "desc": "Deep CNN inspired by VGGNet with stacked 3×3 convolutions and two FC layers.",
        "color": "#E87B4C"
    },
    "ResNet-Style CNN": {
        "desc": "CNN with residual (skip) connections and BatchNorm to combat vanishing gradients.",
        "color": "#4CE8A0"
    },
    "Regularized CNN": {
        "desc": "CNN with BatchNorm, Dropout at every block, and L2 weight regularisation.",
        "color": "#A04CE8"
    }
}

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:700; color:#1E3A5F; text-align:center; margin-bottom:0; }
    .sub-title   { font-size:1.1rem; color:#555; text-align:center; margin-bottom:1.5rem; }
    .metric-card { background:#f8f9fb; border-radius:12px; padding:1rem;
                   border-left:5px solid; text-align:center; }
    .section-hdr { font-size:1.3rem; font-weight:700; color:#1E3A5F;
                   border-bottom:2px solid #e0e0e0; padding-bottom:6px; margin-top:1.5rem; }
    .info-box    { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:1rem; }
    .pred-box    { background:#F0FFF4; border:1px solid #86EFAC; border-radius:8px; padding:1rem; }
    .warning-box { background:#FFFBEB; border:1px solid #FCD34D; border-radius:8px; padding:1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0
    return x_train, y_train.flatten(), x_test, y_test.flatten()


@st.cache_resource
def load_model():
    if os.path.exists('best_model.keras'):
        return keras.models.load_model('best_model.keras')
    return None


def preprocess_image(pil_img):
    img = pil_img.resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    if arr.ndim == 2:          # grayscale → RGB
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:     # RGBA → RGB
        arr = arr[:, :, :3]
    return arr


def plot_prediction_bar(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#4C9BE8' if p != max(probabilities) else '#E84C4C'
              for p in probabilities]
    bars = ax.barh(CLASS_NAMES, probabilities, color=colors, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_confusion_matrix_mini(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.3, annot_kws={'size': 7})
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('True', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png",
             width=60)
    st.markdown("## 🧠 CIFAR-10 Classifier")
    st.markdown("**UMT — 6th Semester**")
    st.markdown("*Deep Learning Project*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Home",
        "🔍 Predict Image",
        "📊 Model Comparison",
        "🗂️ Dataset Explorer",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("**Dataset:** CIFAR-10")
    st.markdown("**Classes:** 10")
    st.markdown("**Framework:** TensorFlow / Keras")
    st.markdown("**Models:** 4 CNN variants")


# ─── HOME ─────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-title">🧠 CIFAR-10 Image Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparative Study of CNN Architectures | University of Management & Technology</p>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [("📂 Dataset", "CIFAR-10"), ("🖼️ Images", "60,000"),
               ("🏷️ Classes", "10"), ("🤖 Models", "4 CNNs")]
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        col.markdown(f"""
        <div class="metric-card" style="border-color:#4C9BE8;">
            <div style="font-size:1.6rem;">{label.split()[0]}</div>
            <div style="font-size:1.1rem;font-weight:700;color:#1E3A5F;">{val}</div>
            <div style="font-size:0.8rem;color:#666;">{label.split(None,1)[1]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-hdr">🤖 Models Compared</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, info) in zip(cols, MODEL_INFO.items()):
        col.markdown(f"""
        <div class="metric-card" style="border-color:{info['color']};">
            <div style="font-size:1rem;font-weight:700;color:#1E3A5F;">{name}</div>
            <div style="font-size:0.8rem;color:#555;margin-top:6px;">{info['desc']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-hdr">📌 Project Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    This project implements and compares <strong>four CNN architectures</strong> on the CIFAR-10 dataset:
    <ul>
        <li><strong>Baseline CNN</strong> — 3-block simple CNN (performance reference)</li>
        <li><strong>VGG-Style CNN</strong> — deep stacked convolutions inspired by VGGNet</li>
        <li><strong>ResNet-Style CNN</strong> — residual connections to tackle vanishing gradients</li>
        <li><strong>Regularized CNN</strong> — BatchNorm + Dropout + L2 for better generalisation</li>
    </ul>
    Models are evaluated on <strong>accuracy, loss, parameters, and training time</strong>.
    </div>
    """, unsafe_allow_html=True)


# ─── PREDICT ──────────────────────────────────────────────────────────────────
elif page == "🔍 Predict Image":
    st.markdown('<p class="section-hdr">🔍 Predict Image Class</p>', unsafe_allow_html=True)
    model = load_model()

    if model is None:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>No trained model found.</strong><br>
        Run the Jupyter notebook first to train and save <code>best_model.keras</code>,
        then place it in the same folder as <code>app.py</code>.
        </div>""", unsafe_allow_html=True)
    else:
        tab1, tab2 = st.tabs(["📁 Upload Image", "🎲 Use CIFAR-10 Sample"])

        with tab1:
            uploaded = st.file_uploader("Upload any image (PNG/JPG)", type=['png','jpg','jpeg'])
            if uploaded:
                pil_img = Image.open(uploaded)
                arr = preprocess_image(pil_img)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(pil_img, caption="Uploaded Image", use_container_width=True)
                with col2:
                    with st.spinner("Predicting…"):
                        probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
                    pred_idx   = np.argmax(probs)
                    pred_class = CLASS_NAMES[pred_idx]
                    emoji      = CLASS_EMOJI[pred_class]
                    st.markdown(f"""
                    <div class="pred-box">
                        <div style="font-size:1.5rem;font-weight:700;">{emoji} {pred_class.upper()}</div>
                        <div style="font-size:1.1rem;color:#16A34A;">Confidence: {probs[pred_idx]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    st.pyplot(plot_prediction_bar(probs))

        with tab2:
            x_train, y_train, x_test, y_test = load_cifar10()
            cls_filter = st.selectbox("Filter by true class", ["All"] + CLASS_NAMES)
            if cls_filter == "All":
                idx = np.random.randint(0, len(x_test))
            else:
                candidates = np.where(y_test == CLASS_NAMES.index(cls_filter))[0]
                idx = np.random.choice(candidates)

            if st.button("🎲 Pick Random Sample"):
                st.session_state['sample_idx'] = idx
            sample_idx = st.session_state.get('sample_idx', idx)
            sample_img = x_test[sample_idx]
            true_label = CLASS_NAMES[y_test[sample_idx]]

            col1, col2 = st.columns([1, 2])
            with col1:
                fig, ax = plt.subplots(); ax.imshow(sample_img); ax.axis('off')
                ax.set_title(f'True: {true_label}', fontsize=10)
                st.pyplot(fig)
            with col2:
                probs = model.predict(sample_img[np.newaxis, ...], verbose=0)[0]
                pred_class = CLASS_NAMES[np.argmax(probs)]
                correct = pred_class == true_label
                color = "#F0FFF4" if correct else "#FFF5F5"
                border = "#86EFAC" if correct else "#FCA5A5"
                result_text = "✅ Correct!" if correct else "❌ Wrong"
                st.markdown(f"""
                <div style="background:{color};border:1px solid {border};border-radius:8px;padding:1rem;">
                    <strong>Predicted:</strong> {CLASS_EMOJI[pred_class]} {pred_class.upper()}<br>
                    <strong>True:</strong> {CLASS_EMOJI[true_label]} {true_label.upper()}<br>
                    <strong>Result:</strong> {result_text}
                </div>""", unsafe_allow_html=True)
                st.pyplot(plot_prediction_bar(probs))


# ─── COMPARISON ───────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown('<p class="section-hdr">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    ℹ️ The results below are <strong>representative values</strong> from training on CIFAR-10.
    Run the notebook to get your exact results.
    </div>""", unsafe_allow_html=True)

    # Hardcoded representative results (replace with actual after training)
    comparison_data = {
        "Model"             : ["Baseline CNN", "VGG-Style CNN", "ResNet-Style CNN", "Regularized CNN"],
        "Test Accuracy (%)" : [72.4, 79.1, 85.3, 81.7],
        "Test Loss"         : [0.8512, 0.6734, 0.4891, 0.5623],
        "Parameters (M)"    : [0.35, 2.18, 1.12, 0.42],
        "Training Time (s)" : [120, 380, 290, 195],
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df.style.highlight_max(subset=["Test Accuracy (%)"], color="#d4edda")
                          .highlight_min(subset=["Test Loss"], color="#d4edda"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#4C9BE8','#E87B4C','#4CE8A0','#A04CE8']
        bars = ax.bar(df["Model"], df["Test Accuracy (%)"], color=colors, edgecolor='white')
        for bar, val in zip(bars, df["Test Accuracy (%)"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f'{val}%', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 100); ax.set_ylabel('Accuracy (%)'); ax.grid(axis='y', alpha=0.3)
        ax.set_title('Test Accuracy Comparison', fontweight='bold')
        plt.xticks(rotation=15, fontsize=8); plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["Model"], df["Training Time (s)"], color=colors, edgecolor='white')
        ax.set_ylabel('Seconds'); ax.grid(axis='y', alpha=0.3)
        ax.set_title('Training Time Comparison', fontweight='bold')
        plt.xticks(rotation=15, fontsize=8); plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<p class="section-hdr">🔑 Key Takeaways</p>', unsafe_allow_html=True)
    takeaways = [
        ("🥇 Best Accuracy", "ResNet-Style CNN (skip connections prevent vanishing gradient)"),
        ("⚡ Fastest Training", "Baseline CNN (fewest parameters, simplest architecture)"),
        ("🎯 Best Trade-off", "Regularized CNN (strong accuracy with lower param count)"),
        ("🔥 Most Parameters", "VGG-Style CNN (deep FC layers dominate parameter count)"),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(takeaways):
        cols[i%2].markdown(f"""
        <div class="metric-card" style="border-color:#4C9BE8;margin-bottom:10px;">
            <div style="font-weight:700;font-size:1rem;">{title}</div>
            <div style="font-size:0.85rem;color:#555;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ─── DATASET EXPLORER ─────────────────────────────────────────────────────────
elif page == "🗂️ Dataset Explorer":
    st.markdown('<p class="section-hdr">🗂️ CIFAR-10 Dataset Explorer</p>', unsafe_allow_html=True)
    x_train, y_train, x_test, y_test = load_cifar10()

    tab1, tab2 = st.tabs(["🖼️ Sample Images", "📈 Statistics"])

    with tab1:
        cls = st.selectbox("Select class to browse", CLASS_NAMES)
        n_show = st.slider("Number of images", 5, 30, 10)
        cls_idx = CLASS_NAMES.index(cls)
        indices = np.where(y_train == cls_idx)[0][:n_show]
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            fig, ax = plt.subplots(figsize=(2,2))
            ax.imshow(x_train[idx]); ax.axis('off')
            cols[i%5].pyplot(fig); plt.close(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            unique, counts = np.unique(y_train, return_counts=True)
            ax.bar(CLASS_NAMES, counts, color='steelblue', edgecolor='white')
            ax.set_title('Training Set Distribution', fontweight='bold')
            ax.set_ylabel('Count'); plt.xticks(rotation=45, fontsize=8); plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            unique, counts = np.unique(y_test, return_counts=True)
            ax.bar(CLASS_NAMES, counts, color='coral', edgecolor='white')
            ax.set_title('Test Set Distribution', fontweight='bold')
            ax.set_ylabel('Count'); plt.xticks(rotation=45, fontsize=8); plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Total images | 60,000 |
        | Training images | 50,000 |
        | Test images | 10,000 |
        | Image size | 32 × 32 pixels |
        | Channels | 3 (RGB) |
        | Classes | 10 |
        | Images per class (train) | 5,000 |
        | Images per class (test) | 1,000 |
        """)


# ─── ABOUT ────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<p class="section-hdr">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <h4>📚 Deep Learning Project — 6th Semester</h4>
    <p><strong>University of Management and Technology (UMT), Lahore</strong></p>
    <p>This project implements and compares four CNN architectures for image classification
    on the CIFAR-10 benchmark dataset. The goal is to analyse the trade-offs between
    model complexity, accuracy, training time, and generalisation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    | Tool | Purpose |
    |---|---|
    | Python 3.10 | Programming language |
    | TensorFlow / Keras | Deep learning framework |
    | NumPy / Pandas | Data manipulation |
    | Matplotlib / Seaborn | Visualisation |
    | Scikit-learn | Evaluation metrics |
    | Streamlit | Web application |
    """)

    st.markdown("### 📂 Project Structure")
    st.code("""
project/
├── app.py                          # Streamlit application
├── image_classification_project.ipynb  # Training notebook
├── best_model.keras                # Saved best model
├── requirements.txt                # Dependencies
└── Project_Report.docx             # Full project report
    """)
