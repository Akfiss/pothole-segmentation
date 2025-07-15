import streamlit as st 
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import tempfile
import os
import pandas as pd

# --- Fungsi untuk Membuat Laporan PDF ---
def create_detection_report_pdf(report_data, summary_image_path=None, model_path_display="N/A", logo_path_display=None):
    """Membuat laporan PDF dari data deteksi."""
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    # Mengurangi margin atas dan bawah sedikit untuk memberi lebih banyak ruang
    doc = SimpleDocTemplate(buffer.name, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name=' 작은 ', fontSize=8, leading=10, alignment=1)) 
    styles['h1'].alignment = 1 
    styles['h1'].fontSize = 18 # Sedikit perbesar judul utama
    styles['h2'].fontSize = 14 # Sedikit perbesar subjudul
    styles['h2'].spaceBefore = 0.2*inch
    styles['h2'].spaceAfter = 0.1*inch
    # Tambahkan style Italic jika belum ada (untuk catatan tabel)
    if 'Italic' not in styles:
        styles.add(ParagraphStyle(name='Italic', parent=styles['Normal'], fontName='Helvetica-Oblique'))

    story = []

    if logo_path_display and os.path.exists(logo_path_display):
        try:
            logo = ReportLabImage(logo_path_display, width=1.2*inch, height=0.6*inch) 
            logo.hAlign = 'LEFT'
            story.append(logo)
            story.append(Spacer(1, 0.05 * inch))
        except Exception as e:
            print(f"Error adding logo to PDF: {e}")

    title = Paragraph("Laporan Analisis Deteksi Lubang Jalan", styles['h1'])
    story.append(title)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(f"<b>Tanggal Laporan:</b> {datetime.now().strftime('%d %B %Y, %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Model Digunakan:</b> YOLOv8-seg", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence Threshold:</b> {report_data['confidence_threshold']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>IoU Threshold:</b> {report_data['iou_threshold']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Referensi Skala:</b> {report_data['pixels_per_meter']} piksel/meter", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Ringkasan Statistik Deteksi:", styles['h2']))
    summary_data = [
        ["Total Lubang Unik Terdeteksi:", f"{report_data['total_unique_potholes']}"],
        ["Total Estimasi Luas Lubang Unik:", f"{report_data['total_new_area_session']:.3f} m²"],
    ]
    avg_area_new = report_data.get('avg_area_new')
    max_area_new = report_data.get('max_area_new')
    min_area_new = report_data.get('min_area_new')

    if avg_area_new is not None:
        summary_data.append(["Rata-rata Luas Lubang Baru:", f"{avg_area_new:.3f} m²"])
    if max_area_new is not None:
        summary_data.append(["Luas Lubang Baru Terbesar:", f"{max_area_new:.3f} m²"])
    if min_area_new is not None:
        summary_data.append(["Luas Lubang Baru Terkecil:", f"{min_area_new:.3f} m²"])
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), 
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), # Vertikal align tengah
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    if summary_image_path and os.path.exists(summary_image_path):
        try:
            story.append(Paragraph("Visualisasi Data Deteksi:", styles['h2']))
            # Sesuaikan lebar gambar agar judul tidak terpotong, mungkin perlu sedikit lebih kecil
            img = ReportLabImage(summary_image_path, width=7*inch, height=3.5*inch) # Sedikit lebih lebar
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            print(f"Error adding image to PDF: {e}")

    if not report_data['df_all_detections'].empty:
        story.append(Paragraph("Detail Deteksi per Frame (Contoh):", styles['h2']))
        
        df_for_report = report_data['df_all_detections'][["frame", "confidence", "area_m2"]].copy() # Buat salinan untuk modifikasi
        
        # Format kolom 'confidence' menjadi persentase
        df_for_report['confidence'] = (df_for_report['confidence'] * 100).round(1).astype(str) + '%'
        
        # Format kolom 'area_m2' dengan satuan dan pembulatan
        df_for_report['area_m2'] = df_for_report['area_m2'].round(3).astype(str) + ' m²'
        
        # Ubah nama kolom
        df_for_report.columns = ["Frame", "Confidence", "Luas"]
        
        # Ambil 20 baris pertama
        df_report_sample = df_for_report.head(20) 

        data_for_table = [df_report_sample.columns.to_list()] + df_report_sample.values.tolist() # Tidak perlu .astype(str) lagi karena sudah string
        
        # Sesuaikan lebar kolom jika perlu
        table = Table(data_for_table, colWidths=[0.7*inch, 1.5*inch, 1.5*inch]) # Disesuaikan untuk 3 kolom
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F8BFF")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#EFF5FF")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0,1), (-1,-1), 8),
        ]))
        story.append(table)
        if len(report_data['df_all_detections']) > 20:
            story.append(Paragraph(f"<i>(Menampilkan 20 baris pertama dari total {len(report_data['df_all_detections'])} deteksi)</i>", styles['Italic']))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Penjelasan Hasil Deteksi:", styles['h2']))
    explanation = f"""
    Laporan ini merangkum hasil deteksi lubang jalan dari analisis video menggunakan model YOLOv8-seg.
    Parameter utama yang digunakan dalam sesi deteksi ini adalah: ambang batas keyakinan sebesar 
    {report_data['confidence_threshold']:.2f} dan ambang batas IoU (Intersection over Union) sebesar {report_data['iou_threshold']:.2f}.
    Referensi skala yang digunakan untuk estimasi luas adalah {report_data['pixels_per_meter']} piksel per meter.
    <br/><br/>
    <b>Total Lubang Unik Terdeteksi</b> menunjukkan jumlah lubang berbeda yang berhasil diidentifikasi dan dilacak selama analisis.
    <b>Total Estimasi Luas Lubang Unik</b> adalah akumulasi luas dari lubang-lubang unik tersebut.
    Statistik rata-rata, terbesar, dan terkecil merujuk pada luas lubang yang teridentifikasi sebagai 'baru' (pertama kali terdeteksi dalam sesi ini).
    <br/><br/>
    Akurasi pengukuran luas sangat bergantung pada ketepatan kalibrasi referensi skala (piksel per meter) 
    terhadap kondisi video atau tangkapan gambar yang dianalisis. Perbedaan perspektif, jarak kamera, 
    dan resolusi dapat memengaruhi hasil jika skala tidak disesuaikan dengan benar untuk setiap kondisi.
    """
    story.append(Paragraph(explanation, styles['Normal']))
    
    story.append(Spacer(1, 1 * inch))
    story.append(Paragraph(f"Laporan dihasilkan oleh Aplikasi Deteksi Lubang Jalan", styles[' 작은 '])) 
    story.append(Paragraph(f"© {datetime.now().year} Akbar Johan Firdaus - Universitas Udayana", styles[' 작은 ']))

    doc.build(story)
    return buffer.name