import streamlit as st
from ultralytics import YOLO

@st.cache_resource # Cache model agar tidak di-load ulang setiap interaksi
def load_yolo_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(model_path)
        # Tidak perlu st.success di sini karena akan dipanggil setiap rerun jika tidak di-cache
        # Cukup kembalikan model atau None
        return model
    except Exception as e:
        # Tampilkan error di tempat yang lebih sesuai (misal di main_app.py saat pemanggilan)
        print(f"Error saat memuat model dari '{model_path}': {e}")
        print("Pastikan file model ada di direktori yang benar atau path sudah sesuai.")
        return None