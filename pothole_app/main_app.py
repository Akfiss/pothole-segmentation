# main_app.py

import streamlit as st
import tempfile
import os
import cv2
from datetime import datetime
import pandas as pd
import numpy as np # <-- Tambahkan import ini

# Impor dari file-file modular
from model_loader import load_yolo_model
from frame_processor import process_and_draw_frame
from ui_components import setup_sidebar, display_summary_and_export, update_sidebar_stats, add_reset_button 

# --- Konfigurasi Aplikasi & Pemuatan Model ---
MODEL_PATH = 'pothole_app/best.pt' # Pastikan path ini benar
LOGO_PATH = None 

st.set_page_config(
    page_title="Deteksi & Pengukuran Lubang Jalan",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = load_yolo_model(MODEL_PATH)
if model is None: 
    st.error("GAGAL MEMUAT MODEL. Pastikan path model benar dan file model tidak korup. Aplikasi mungkin tidak berfungsi dengan benar.")

# --- Inisialisasi Session State (jika belum ada) ---
DEFAULT_SESSION_VALUES = {
    'confidence_threshold': 0.6, 
    'iou_threshold': 0.5, 
    'pixels_per_meter': 300,
    'show_boxes_opt': True, 
    'box_color_hex_val': "#FF0000", 
    'show_masks_opt': True,
    'webcam_running': False, 
    'tracked_potholes_session': [], 
    'total_new_area_session': 0.0,
    'all_session_detections_details': [], 
    'summary_displayed_after_webcam': False, 
    'frame_count_webcam': 0, 
    'current_video_processing_done': False, 
    'current_webcam_session_done': False, 
    'uploaded_file_key': 0, 
    'app_mode': "Unggah Gambar", # <-- Ubah default ke mode baru
    'selected_camera_index': 0,
    # BARU: State untuk mode gambar
    'current_image_processing_done': False,
    'processed_image_to_display': None,
    'image_detection_details': [],
    'uploaded_image_key': 100
}

for key, value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Inisialisasi box_color_bgr_val jika belum ada (dilakukan di setup_sidebar)

st.title("🚧 Deteksi & Pengukuran Lubang Jalan 🚧")
st.markdown("""
Aplikasi ini menggunakan model YOLOv8-seg untuk mendeteksi dan melakukan segmentasi pada lubang jalan. 
Anda dapat mengunggah gambar, video, atau menggunakan webcam untuk deteksi.
""")
st.markdown("---") 

setup_sidebar() # Ini akan menginisialisasi atau mengupdate nilai slider di session_state

st.markdown("---") 
col_mode_select, col_mode_info = st.columns([1,2]) 
with col_mode_select:
    # BARU: Tambahkan 'Unggah Gambar' ke daftar mode
    mode_options = ["Unggah Gambar", "Unggah Video", "Deteksi Real-time (Webcam)"]
    current_mode_index = mode_options.index(st.session_state.app_mode)
    app_mode_selected_value = st.selectbox("Pilih Mode Deteksi:", 
                                           mode_options, 
                                           index=current_mode_index,
                                           key="app_mode_select_main_v6") 

if app_mode_selected_value != st.session_state.app_mode:
    st.session_state.app_mode = app_mode_selected_value 
    # Reset semua state yang relevan saat berganti mode
    st.session_state.webcam_running = False
    st.session_state.current_video_processing_done = False
    st.session_state.current_webcam_session_done = False
    st.session_state.current_image_processing_done = False # <-- Reset state gambar
    st.session_state.all_session_detections_details = []
    st.session_state.tracked_potholes_session = []
    st.session_state.total_new_area_session = 0.0
    st.session_state.summary_displayed_after_webcam = False
    st.session_state.frame_count_webcam = 0
    update_sidebar_stats()
    st.rerun() 

with col_mode_info:
    if st.session_state.app_mode == "Unggah Gambar":
        st.info("Mode Unggah Gambar: Deteksi lubang pada satu gambar.")
    elif st.session_state.app_mode == "Unggah Video":
        st.info("Mode Unggah Video: Proses video yang sudah ada untuk analisis mendalam.")
    else:
        st.info(f"Mode Webcam: Deteksi langsung menggunakan kamera (Indeks: {st.session_state.selected_camera_index}).")
st.markdown("---") 

main_output_container = st.container()

# ==============================================================================
# --- BARU: LOGIKA UNTUK MODE UNGGAH GAMBAR ---
# ==============================================================================
if st.session_state.app_mode == "Unggah Gambar":
    with main_output_container:
        st.header("🖼️ Unggah Gambar untuk Deteksi")
        uploaded_image_file = st.file_uploader(
            "Pilih file gambar...",
            type=["jpg", "jpeg", "png", "bmp"],
            key=f"image_uploader_main_v6_{st.session_state.uploaded_image_key}"
        )

        start_detection_button_image = None
        if uploaded_image_file is not None and model is not None:
            start_detection_button_image = st.button("🚀 Mulai Deteksi Gambar", key="start_image_button_main_v6", help="Klik untuk memulai proses analisis gambar.", use_container_width=True)

        if start_detection_button_image and uploaded_image_file is not None and model is not None:
            st.session_state.current_image_processing_done = False
            st.session_state.image_detection_details = []
            
            # Baca gambar menggunakan OpenCV
            file_bytes = np.asarray(bytearray(uploaded_image_file.read()), dtype=np.uint8)
            source_image = cv2.imdecode(file_bytes, 1) # 1 = cv2.IMREAD_COLOR
            
            with st.spinner("Memproses gambar..."):
                try:
                    # Gunakan kembali fungsi process_and_draw_frame
                    annotated_image, pothole_details, _ = process_and_draw_frame(
                        source_image, model,
                        st.session_state.confidence_threshold,
                        st.session_state.iou_threshold, # NMS, tidak untuk tracking
                        st.session_state.pixels_per_meter,
                        st.session_state.show_boxes_opt,
                        st.session_state.box_color_bgr_val,
                        st.session_state.show_masks_opt,
                        tracked_potholes_session_bboxes=[], # Tidak ada tracking untuk satu gambar
                        update_tracked_list=False # Tidak ada tracking
                    )
                    
                    st.session_state.processed_image_to_display = annotated_image
                    st.session_state.image_detection_details = pothole_details
                    st.session_state.current_image_processing_done = True

                except Exception as e:
                    st.error(f"Error saat memproses gambar: {e}")
                    st.session_state.current_image_processing_done = False

        if st.session_state.get('current_image_processing_done', False):
            st.subheader("✔️ Hasil Deteksi Gambar")
            
            # Tampilkan gambar yang sudah diproses
            processed_img_rgb = cv2.cvtColor(st.session_state.processed_image_to_display, cv2.COLOR_BGR2RGB)
            st.image(processed_img_rgb, caption="Gambar Hasil Deteksi", use_container_width=True)

            # Tampilkan statistik sederhana
            total_potholes_img = len(st.session_state.image_detection_details)
            total_area_img = sum(p['area_m2'] for p in st.session_state.image_detection_details)

            st.metric("Jumlah Lubang Terdeteksi", f"{total_potholes_img}")
            st.metric("Total Estimasi Luas", f"{total_area_img:.3f} m²")

            # Tombol untuk mengunduh gambar
            _, buf = cv2.imencode('.png', st.session_state.processed_image_to_display)
            img_bytes = buf.tobytes()
            
            st.download_button(
                label="📥 Unduh Gambar Hasil Deteksi",
                data=img_bytes,
                file_name=f"hasil_deteksi_{os.path.splitext(uploaded_image_file.name)[0]}.png",
                mime="image/png",
                use_container_width=True
            )
            
            with st.expander("Lihat Detail Deteksi"):
                st.dataframe(pd.DataFrame(st.session_state.image_detection_details))
            
            add_reset_button() # Tombol reset sesi

        elif uploaded_image_file is None and model is not None:
            st.info("Silakan unggah file gambar dan klik 'Mulai Deteksi Gambar'.")
        elif model is None:
            st.warning("Model belum dimuat. Fitur unggah gambar tidak dapat digunakan.")


# ==============================================================================
# --- LOGIKA UNTUK MODE UNGGAH VIDEO (TETAP SAMA) ---
# ==============================================================================
elif st.session_state.app_mode == "Unggah Video":
    with main_output_container: 
        st.header("🎞️ Unggah Video untuk Deteksi")
        uploaded_file = st.file_uploader("Pilih file video...", 
                                         type=["mp4", "avi", "mov", "mkv"], 
                                         key=f"video_uploader_main_v5_{st.session_state.uploaded_file_key}")

        start_detection_button_upload = None
        if uploaded_file is not None and model is not None:
            start_detection_button_upload = st.button("🚀 Mulai Deteksi Video", key="start_video_button_main_v5", help="Klik untuk memulai proses analisis video.", use_container_width=True)

        frame_display_placeholder_upload = st.empty()

        if start_detection_button_upload and uploaded_file is not None and model is not None:
            st.session_state.current_video_processing_done = False 
            st.session_state.all_session_detections_details = []   
            st.session_state.tracked_potholes_session = []
            st.session_state.total_new_area_session = 0.0
            update_sidebar_stats()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path_upload = tmp_file.name
            
            try:
                cap = cv2.VideoCapture(video_path_upload)
                if not cap.isOpened(): 
                    st.error("Error: Tidak dapat membuka file video.")
                else:
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS)); 
                    if fps == 0: fps = 30 

                    output_video_path_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    out_writer = cv2.VideoWriter(output_video_path_temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                    
                    progress_bar_video = st.progress(0, text="Memulai pemrosesan video...")
                    st.info("Harap tunggu, video sedang diproses...")
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0: 
                        st.warning("Tidak dapat membaca total frame video. Progress bar mungkin tidak akurat.")
                    frame_count_video = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count_video += 1
                        
                        annotated_frame, frame_potholes_info, newly_detected_area = process_and_draw_frame(
                            frame, model, st.session_state.confidence_threshold, st.session_state.iou_threshold, 
                            st.session_state.pixels_per_meter, st.session_state.show_boxes_opt, 
                            st.session_state.box_color_bgr_val, st.session_state.show_masks_opt, 
                            tracked_potholes_session_bboxes=st.session_state.tracked_potholes_session, 
                            update_tracked_list=True
                        )
                        st.session_state.total_new_area_session += newly_detected_area
                        
                        for pothole in frame_potholes_info: pothole["frame"] = frame_count_video
                        st.session_state.all_session_detections_details.extend(frame_potholes_info)
                        
                        out_writer.write(annotated_frame)
                        frame_display_placeholder_upload.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        
                        if total_frames > 0 : 
                            progress_bar_video.progress(frame_count_video / total_frames, text=f"Memproses Frame {frame_count_video}/{total_frames}")
                        else: 
                            progress_bar_video.progress(0, text=f"Memproses Frame {frame_count_video}")
                        update_sidebar_stats()

                    cap.release()
                    out_writer.release()
                    
                    progress_bar_video.empty() 
                    st.success("Video unggahan berhasil diproses!")
                    st.session_state.current_video_processing_done = True 
                    
                    st.session_state.processed_video_path = output_video_path_temp
                    st.session_state.original_video_name_for_download = uploaded_file.name
                    st.rerun()

            except Exception as e: 
                st.error(f"Error pemrosesan video unggahan: {e}")
                st.session_state.current_video_processing_done = False
            finally:
                if 'video_path_upload' in locals() and os.path.exists(video_path_upload): 
                    os.remove(video_path_upload)

        if st.session_state.get('current_video_processing_done', False):
            frame_display_placeholder_upload.empty() 
            if 'processed_video_path' in st.session_state and os.path.exists(st.session_state.processed_video_path):
                with open(st.session_state.processed_video_path, "rb") as file_processed:
                    st.download_button(label="Unduh Video Hasil Deteksi", 
                                       data=file_processed,
                                       file_name=f"hasil_deteksi_{st.session_state.get('original_video_name_for_download', 'video.mp4')}", 
                                       mime="video/mp4", 
                                       key="download_video_button_main_v5") 
            
            display_summary_and_export("Video Unggahan", MODEL_PATH, LOGO_PATH)
            add_reset_button() 

        elif uploaded_file is None and model is not None:
            st.info("Silakan unggah file video dan klik 'Mulai Deteksi Video'.")
        elif model is None:
            st.warning("Model belum dimuat. Fitur unggah video tidak dapat digunakan.")


# ==============================================================================
# --- LOGIKA UNTUK MODE WEBCAM (TETAP SAMA) ---
# ==============================================================================
elif st.session_state.app_mode == "Deteksi Real-time (Webcam)":
    with main_output_container:
        st.header("📷 Deteksi Real-time via Webcam")
        if model is None:
            st.warning("Model belum dimuat. Deteksi webcam tidak dapat dimulai.")
        else:
            webcam_control_cols = st.columns([1,1,2]) 
            with webcam_control_cols[0]:
                if st.button("▶️ Mulai Webcam", key="start_webcam_button_main_v5", disabled=st.session_state.webcam_running, use_container_width=True):
                    st.session_state.webcam_running = True
                    st.session_state.tracked_potholes_session = [] 
                    st.session_state.total_new_area_session = 0.0
                    st.session_state.all_session_detections_details = [] 
                    st.session_state.summary_displayed_after_webcam = False 
                    st.session_state.frame_count_webcam = 0 
                    st.session_state.current_webcam_session_done = False 
                    update_sidebar_stats()
                    st.rerun() 
            with webcam_control_cols[1]:
                if st.button("⏹️ Hentikan Webcam", key="stop_webcam_button_main_v5", disabled=not st.session_state.webcam_running, use_container_width=True):
                    st.session_state.webcam_running = False
                    st.session_state.current_webcam_session_done = True 
                    st.rerun() 
            with webcam_control_cols[2]:
                if st.session_state.webcam_running:
                    st.success(f"Webcam Aktif (Indeks: {st.session_state.selected_camera_index})...") # Tampilkan indeks yang digunakan
                elif st.session_state.current_webcam_session_done:
                    st.info("Webcam Tidak Aktif. Ringkasan sesi terakhir ditampilkan.")
                else:
                    st.info("Webcam Tidak Aktif.")

            stframe_webcam_placeholder = st.empty() 

            if st.session_state.webcam_running:
                # Gunakan indeks kamera dari session_state
                camera_to_use = int(st.session_state.get('selected_camera_index', 0))
                cap_webcam = cv2.VideoCapture(camera_to_use) 
                
                if not cap_webcam.isOpened():
                    st.error(f"Error: Tidak dapat mengakses webcam dengan indeks {camera_to_use}. Coba indeks lain atau pastikan kamera tersedia.")
                    st.session_state.webcam_running = False 
                    st.rerun() 
                else:
                    while st.session_state.webcam_running: 
                        ret, frame = cap_webcam.read()
                        if not ret:
                            st.warning("Gagal membaca frame dari webcam.")
                            break
                        
                        st.session_state.frame_count_webcam += 1
                        
                        annotated_frame, frame_potholes_info, newly_detected_area_webcam = process_and_draw_frame(
                            frame, model, st.session_state.confidence_threshold, st.session_state.iou_threshold, 
                            st.session_state.pixels_per_meter, st.session_state.show_boxes_opt, 
                            st.session_state.box_color_bgr_val, st.session_state.show_masks_opt,
                            tracked_potholes_session_bboxes=st.session_state.tracked_potholes_session, 
                            update_tracked_list=True
                        )
                        st.session_state.total_new_area_session += newly_detected_area_webcam
                        
                        for pothole in frame_potholes_info: 
                            pothole["frame"] = st.session_state.frame_count_webcam
                        st.session_state.all_session_detections_details.extend(frame_potholes_info)
                        
                        stframe_webcam_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        update_sidebar_stats()
                    
                    cap_webcam.release()
            
            if st.session_state.current_webcam_session_done: 
                stframe_webcam_placeholder.empty() 
                if st.session_state.all_session_detections_details:
                    display_summary_and_export("Webcam", MODEL_PATH, LOGO_PATH)
                    add_reset_button() 
                else:
                    st.info("Tidak ada lubang terdeteksi selama sesi webcam ini.")
                    add_reset_button() 
                st.session_state.summary_displayed_after_webcam = True 
            
            elif not st.session_state.webcam_running and not st.session_state.all_session_detections_details:
                stframe_webcam_placeholder.empty()
