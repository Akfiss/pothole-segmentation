import streamlit as st # Meskipun tidak digunakan langsung di sini, umum untuk ada
import numpy as np

def hex_to_bgr(hex_color):
    """Konversi warna hex ke tuple BGR."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    # Pastikan lv adalah kelipatan 3 (untuk R, G, B)
    if lv == 3: # Format singkat seperti #RGB
        hex_color = "".join([c*2 for c in hex_color])
        lv = 6
    if lv != 6:
        # Fallback ke warna default jika format hex tidak valid
        print(f"Format warna hex tidak valid: {hex_color}, menggunakan hitam sebagai default.")
        return (0,0,0) # Hitam
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))[::-1]

def is_new_pothole(new_bbox_coords, tracked_potholes_bboxes, iou_threshold=0.5):
    """
    Periksa apakah lubang yang terdeteksi baru berdasarkan Intersection over Union (IoU)
    dengan lubang yang sudah dilacak dalam sesi ini.
    """
    x1_new, y1_new, x2_new, y2_new = new_bbox_coords
    for tracked_bbox in tracked_potholes_bboxes:
        x1_tracked, y1_tracked, x2_tracked, y2_tracked = tracked_bbox
        
        xi1 = max(x1_new, x1_tracked)
        yi1 = max(y1_new, y1_tracked)
        xi2 = min(x2_new, x2_tracked)
        yi2 = min(y2_new, y2_tracked)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        intersection = inter_width * inter_height
        
        area_new = (x2_new - x1_new) * (y2_new - y1_new)
        area_tracked = (x2_tracked - x1_tracked) * (y2_tracked - y1_tracked)
        union = area_new + area_tracked - intersection
        
        if union == 0: 
            iou = 0
        else:
            iou = intersection / union
            
        if iou > iou_threshold:
            return False 
    return True