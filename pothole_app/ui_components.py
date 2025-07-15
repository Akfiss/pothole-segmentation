# ui_components.py

import streamlit as st
import pandas as pd
from utils import hex_to_bgr 
from plot_utils import create_streamlit_summary_chart_data, create_static_summary_plot_for_pdf 
from report_generator import create_detection_report_pdf 
import os
from datetime import datetime

def setup_sidebar():
    """Mengatur dan menampilkan widget di sidebar."""
    st.sidebar.header("⚙️ Pengaturan Deteksi")
    
    # Pengaturan Deteksi Umum
    st.session_state.confidence_threshold = st.sidebar.slider('Confidence Threshold', 
        min_value=0.0, max_value=1.0, value=st.session_state.get('confidence_threshold', 0.6), step=0.05, key="conf_slider_ui_v6")
    st.session_state.iou_threshold = st.sidebar.slider('IoU Threshold (NMS & Tracking)', 
        min_value=0.0, max_value=1.0, value=st.session_state.get('iou_threshold', 0.5), step=0.05, key="iou_slider_ui_v6")
    st.session_state.pixels_per_meter = st.sidebar.slider('Referensi Skala (Piksel per Meter)', 
        min_value=10, max_value=2000, value=st.session_state.get('pixels_per_meter', 300), step=10, key="ppm_slider_ui_v6",
        help="Sesuaikan nilai ini berdasarkan jarak kamera ke objek dan resolusi video untuk akurasi pengukuran luas.")

    # Pengaturan Spesifik Webcam
    st.sidebar.header("📷 Pengaturan Webcam")
    st.session_state.selected_camera_index = st.sidebar.number_input(
        "Indeks Kamera (0 untuk default, coba 1, 2, dst. untuk OBS Virtual Cam)", 
        min_value=0, 
        max_value=10, # Batas atas yang wajar, bisa disesuaikan
        value=st.session_state.get('selected_camera_index', 0), 
        step=1,
        key="camera_index_selector_v6",
        help="Masukkan nomor indeks untuk kamera yang ingin digunakan. Kamera default biasanya 0. OBS Virtual Camera mungkin memiliki indeks lain."
    )


    st.sidebar.header("🎨 Pengaturan Visualisasi")
    st.session_state.show_boxes_opt = st.sidebar.checkbox("Tampilkan Bounding Box", st.session_state.get('show_boxes_opt', True), key="show_box_check_ui_v6")
    
    new_box_color_hex = st.sidebar.color_picker("Warna Bounding Box", st.session_state.get('box_color_hex_val', "#FF0000"), key="box_color_pick_ui_v6")
    if new_box_color_hex != st.session_state.get('box_color_hex_val'):
        st.session_state.box_color_hex_val = new_box_color_hex
        st.session_state.box_color_bgr_val = hex_to_bgr(new_box_color_hex)
    elif 'box_color_bgr_val' not in st.session_state: 
        st.session_state.box_color_bgr_val = hex_to_bgr(st.session_state.get('box_color_hex_val', "#FF0000"))

    st.session_state.show_masks_opt = st.sidebar.checkbox("Tampilkan Mask Segmentasi (via plot())", st.session_state.get('show_masks_opt', True), key="show_mask_check_ui_v6")
    st.sidebar.markdown("_Catatan: Warna & opasitas mask diatur oleh fungsi `plot()` bawaan YOLO._")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Statistik Sesi Video/Webcam") # <-- Judul diubah agar lebih spesifik
    if 'total_new_area_placeholder' not in st.session_state:
        st.session_state.total_new_area_placeholder = st.sidebar.empty()
    if 'total_new_potholes_placeholder' not in st.session_state:
        st.session_state.total_new_potholes_placeholder = st.sidebar.empty()
    update_sidebar_stats() 

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <p style="font-size: 0.9em;">Dikembangkan oleh: <br/><b>Akbar Johan Firdaus</b></p>
            <p style="font-size: 0.8em;">Tugas Akhir - Universitas Udayana</p>
        </div>
        """, unsafe_allow_html=True
    )

def update_sidebar_stats():
    """Mengupdate tampilan statistik di sidebar untuk sesi video/webcam."""
    if hasattr(st.session_state.get('total_new_area_placeholder'), 'markdown'):
        st.session_state.total_new_area_placeholder.markdown(f"**Total Luas Baru:** {st.session_state.get('total_new_area_session', 0.0):.3f} m²")
    if hasattr(st.session_state.get('total_new_potholes_placeholder'), 'markdown'):
        st.session_state.total_new_potholes_placeholder.markdown(f"**Total Lubang Baru:** {len(st.session_state.get('tracked_potholes_session', []))}")

def reset_session_state_values():
    """Mereset nilai-nilai kunci di session state untuk memulai sesi baru."""
    keys_to_reset = [
        # State Video & Webcam
        'tracked_potholes_session', 'total_new_area_session', 
        'all_session_detections_details', 'summary_displayed_after_webcam',
        'frame_count_webcam', 'current_video_processing_done', 
        'current_webcam_session_done', 'processed_video_path',
        'original_video_name_for_download',
        # BARU: State Gambar
        'current_image_processing_done', 'processed_image_to_display',
        'image_detection_details'
    ]
    default_values_for_reset = {
        # Default Video & Webcam
        'tracked_potholes_session': [], 'total_new_area_session': 0.0,
        'all_session_detections_details': [], 'summary_displayed_after_webcam': False,
        'frame_count_webcam': 0, 'current_video_processing_done': False,
        'current_webcam_session_done': False,
        # BARU: Default Gambar
        'current_image_processing_done': False,
        'processed_image_to_display': None,
        'image_detection_details': []
    }

    for key in keys_to_reset:
        if key in st.session_state:
            if key in default_values_for_reset:
                st.session_state[key] = default_values_for_reset[key]
            else: 
                try:
                    if key == 'processed_video_path' and st.session_state.get(key) and os.path.exists(st.session_state[key]):
                        os.remove(st.session_state[key])
                    # Hapus key dari session state jika ada
                    if key in st.session_state:
                        del st.session_state[key]
                except Exception as e:
                    print(f"Error saat mereset key {key}: {e}")
    
    # Reset kunci uploader untuk video dan gambar
    st.session_state.uploaded_file_key = st.session_state.get('uploaded_file_key', 0) + 1
    st.session_state.uploaded_image_key = st.session_state.get('uploaded_image_key', 100) + 1
    update_sidebar_stats()


def add_reset_button():
    """Menambahkan tombol Reset Sesi."""
    if st.button("🔄 Reset & Mulai Baru", key="reset_session_button_v6", help="Klik untuk menghapus hasil saat ini dan memulai deteksi baru.", use_container_width=True):
        reset_session_state_values()
        st.rerun() 

# Fungsi display_summary_and_export tetap sama karena hanya digunakan untuk mode Video dan Webcam
def display_summary_and_export(session_type_name, model_path_display, logo_path_display):
    """Menampilkan ringkasan deteksi, grafik, dan tombol ekspor PDF."""
    if st.session_state.get('all_session_detections_details'):
        st.header(f"📊 Ringkasan Hasil Deteksi {session_type_name}")
        df_session_potholes = pd.DataFrame(st.session_state.all_session_detections_details)
        num_unique_potholes_session = len(st.session_state.tracked_potholes_session)
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.metric("Total Lubang Unik Terdeteksi", num_unique_potholes_session, help="Jumlah lubang berbeda yang terdeteksi dan dilacak dalam sesi ini.")
        with col_sum2:
            st.metric("Total Luas Lubang Unik", f"{st.session_state.total_new_area_session:.3f} m²", help="Akumulasi luas dari lubang-lubang unik yang terdeteksi.")

        df_new_potholes_session = df_session_potholes[df_session_potholes['is_new'] == True]
        if not df_new_potholes_session.empty and 'area_m2' in df_new_potholes_session.columns and not df_new_potholes_session['area_m2'].empty :
            valid_areas = df_new_potholes_session['area_m2'][df_new_potholes_session['area_m2'] > 0]
            avg_area_new = valid_areas.mean() if not valid_areas.empty else None
            max_area_new = df_new_potholes_session['area_m2'].max() if not df_new_potholes_session['area_m2'].empty else None
            min_area_new = valid_areas.min() if not valid_areas.empty else None
            
            st.subheader("Statistik Luas Lubang Baru:")
            col1_res, col2_res, col3_res = st.columns(3)
            col1_res.metric("Rata-rata Luas", f"{avg_area_new:.3f} m²" if avg_area_new is not None and not pd.isna(avg_area_new) else "N/A")
            col2_res.metric("Luas Terbesar", f"{max_area_new:.3f} m²" if max_area_new is not None and not pd.isna(max_area_new) else "N/A")
            col3_res.metric("Luas Terkecil", f"{min_area_new:.3f} m²" if min_area_new is not None and not pd.isna(min_area_new) else "N/A")
        
        streamlit_chart_data = create_streamlit_summary_chart_data(df_new_potholes_session) 
        if streamlit_chart_data is not None and not streamlit_chart_data.empty:
            st.subheader("Grafik Analisis Deteksi Lubang Baru per Frame")
            if 'Jumlah Lubang Baru' in streamlit_chart_data.columns:
                    st.area_chart(streamlit_chart_data['Jumlah Lubang Baru'], use_container_width=True)
            if 'Rata-rata Luas Baru (m²)' in streamlit_chart_data.columns:
                    st.line_chart(streamlit_chart_data['Rata-rata Luas Baru (m²)'], use_container_width=True)
        
        with st.expander("Lihat Detail Semua Deteksi per Frame"):
            st.dataframe(df_session_potholes[["frame", "confidence", "area_m2", "is_new", "x1", "y1", "x2", "y2"]].style.format({
                "confidence": "{:.2f}", "area_m2": "{:.3f}"}))

        st.markdown("---")
        st.subheader("Ekspor Laporan")
        static_plot_path = create_static_summary_plot_for_pdf(df_new_potholes_session)

        report_data_dict = {
            'confidence_threshold': st.session_state.confidence_threshold,
            'iou_threshold': st.session_state.iou_threshold,
            'pixels_per_meter': st.session_state.pixels_per_meter,
            'total_unique_potholes': num_unique_potholes_session,
            'total_new_area_session': st.session_state.total_new_area_session,
            'avg_area_new': avg_area_new if 'avg_area_new' in locals() and avg_area_new is not None and not pd.isna(avg_area_new) else None,
            'max_area_new': max_area_new if 'max_area_new' in locals() and max_area_new is not None and not pd.isna(max_area_new) else None,
            'min_area_new': min_area_new if 'min_area_new' in locals() and min_area_new is not None and not pd.isna(min_area_new) else None,
            'df_all_detections': df_session_potholes
        }
        
        pdf_file_name = f"laporan_deteksi_lubang_{session_type_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        try:
            pdf_path = create_detection_report_pdf(report_data_dict, 
                                                   summary_image_path=static_plot_path, 
                                                   model_path_display=model_path_display, 
                                                   logo_path_display=logo_path_display)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Unduh Laporan PDF", data=pdf_file, file_name=pdf_file_name,
                    mime="application/octet-stream", key=f"pdf_download_btn_{session_type_name}_v6" 
                )
            if os.path.exists(pdf_path): os.remove(pdf_path) 
        except Exception as e:
            st.error(f"Gagal membuat laporan PDF: {e}")
        
        if static_plot_path and os.path.exists(static_plot_path): 
            os.remove(static_plot_path) 

    elif not st.session_state.get('webcam_running', False): 
        st.info(f"Tidak ada lubang terdeteksi selama sesi {session_type_name} ini.")