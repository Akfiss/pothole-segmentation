import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

# --- Fungsi untuk Membuat Data Grafik untuk Streamlit Chart ---
def create_streamlit_summary_chart_data(df_detections):
    """
    Menyiapkan data untuk st.line_chart atau st.bar_chart dari Streamlit.
    Mengembalikan DataFrame yang cocok.
    """
    if df_detections.empty or 'is_new' not in df_detections.columns:
        return None

    df_plot = df_detections[df_detections['is_new']].copy() 
    if df_plot.empty:
        return None

    # Agregasi data per frame
    new_potholes_per_frame = df_plot.groupby('frame').size().reset_index(name='Jumlah Lubang Baru')
    avg_area_new_per_frame = df_plot[df_plot['area_m2'] > 0].groupby('frame')['area_m2'].mean().reset_index(name='Rata-rata Luas Baru (m²)')

    # Gabungkan data untuk plotting
    chart_data = pd.merge(new_potholes_per_frame, avg_area_new_per_frame, on='frame', how='outer').fillna(0)
    
    if chart_data.empty:
        return None
    
    if not chart_data.empty:
        chart_data = chart_data.set_index('frame')

    return chart_data


# --- Fungsi untuk Membuat Grafik Statis (Matplotlib) untuk PDF ---
def create_static_summary_plot_for_pdf(df_detections):
    """Membuat grafik ringkasan statis untuk PDF."""
    if df_detections.empty or 'is_new' not in df_detections.columns:
        return None

    df_plot = df_detections[df_detections['is_new']].copy() 
    if df_plot.empty: return None

    new_potholes_per_frame = df_plot.groupby('frame').size()
    avg_area_new_per_frame = df_plot[df_plot['area_m2'] > 0].groupby('frame')['area_m2'].mean()

    if new_potholes_per_frame.empty and avg_area_new_per_frame.empty: return None

    fig, ax1 = plt.subplots(figsize=(10, 4)) 
    # plt.style.use('seaborn-v0_8-whitegrid') # Bisa dikomentari jika ingin default Matplotlib

    if not new_potholes_per_frame.empty:
        color = 'tab:blue'
        ax1.set_xlabel('Nomor Frame', fontsize=10)
        ax1.set_ylabel('Jumlah Lubang Baru', color=color, fontsize=10)
        ax1.bar(new_potholes_per_frame.index, new_potholes_per_frame.values, color=color, alpha=0.7, label='Lubang Baru')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        ax1.legend(loc='upper left', fontsize=8)

    if not avg_area_new_per_frame.empty:
        ax2 = ax1.twinx() # Membuat sumbu y kedua yang berbagi sumbu x yang sama
        color = 'tab:red'
        ax2.set_ylabel('Rata-rata Luas Baru (m²)', color=color, fontsize=10)
        ax2.plot(avg_area_new_per_frame.index, avg_area_new_per_frame.values, color=color, linestyle='--', marker='.', markersize=5, label='Rata-rata Luas Baru')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=8)
        ax2.legend(loc='upper right', fontsize=8) # Posisikan legenda agar tidak tumpang tindih
    
    fig.tight_layout() # Menyesuaikan layout agar tidak ada elemen yang terpotong
    plt.title('Analisis Deteksi Lubang Baru per Frame', fontsize=12, fontweight='bold')
    
    # Simpan grafik ke file sementara
    plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(plot_path, dpi=200) # DPI lebih tinggi untuk kualitas PDF
    plt.close(fig) # Tutup figure untuk membebaskan memori
    return plot_path