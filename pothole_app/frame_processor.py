import streamlit as st # Untuk st.session_state
import cv2
import numpy as np
from utils import is_new_pothole 

def process_and_draw_frame(frame, yolo_model, confidence_thresh, iou_thresh, pixels_per_meter,
                           show_boxes, box_color_bgr, show_masks,
                           tracked_potholes_session_bboxes=None, update_tracked_list=False):
    """
    Memproses satu frame, melakukan inferensi, menggambar deteksi (mask via plot(), box manual).
    Mengembalikan frame yang telah dianotasi, daftar info lubang, dan area baru.
    """
    if yolo_model is None: 
        return frame, [], 0.0

    results = yolo_model.predict(source=frame, imgsz=640, conf=confidence_thresh, iou=iou_thresh, verbose=False)
    
    if show_masks and results[0].masks is not None:
        annotated_frame = results[0].plot(masks=True, boxes=False, line_width=1) 
    else:
        annotated_frame = frame.copy()
    
    pothole_details_current_frame = []
    newly_detected_area_in_frame = 0.0

    if results[0].boxes is not None:
        for i, box_obj in enumerate(results[0].boxes):
            coords_abs = box_obj.xyxy[0].cpu().numpy().astype(int)
            x1_box, y1_box, x2_box, y2_box = coords_abs
            conf = float(box_obj.conf[0])
            
            current_bbox_coords = (x1_box, y1_box, x2_box, y2_box)
            is_new = True
            if tracked_potholes_session_bboxes is not None:
                # Pastikan st.session_state.iou_threshold ada dan valid
                current_iou_threshold = st.session_state.get('iou_threshold', 0.45) # Default jika tidak ada
                is_new = is_new_pothole(current_bbox_coords, tracked_potholes_session_bboxes, current_iou_threshold) 

            pothole_area_m2 = 0.0
            if results[0].masks is not None and i < len(results[0].masks.data):
                mask_for_area = results[0].masks.data[i].cpu().numpy()
                if (x2_box - x1_box) > 0 and (y2_box - y1_box) > 0: 
                    mask_resized_to_box_for_area = cv2.resize(mask_for_area, (x2_box - x1_box, y2_box - y1_box), interpolation=cv2.INTER_NEAREST)
                    pothole_pixels_in_frame_box = np.sum(mask_resized_to_box_for_area > 0.5)
                    if pixels_per_meter > 0:
                        pothole_area_m2 = pothole_pixels_in_frame_box / (pixels_per_meter ** 2)
            
            pothole_details_current_frame.append({
                "confidence": conf, "area_m2": pothole_area_m2, "is_new": is_new,
                "x1": x1_box, "y1": y1_box, "x2": x2_box, "y2": y2_box
            })

            if is_new and update_tracked_list and tracked_potholes_session_bboxes is not None:
                tracked_potholes_session_bboxes.append(current_bbox_coords)
                newly_detected_area_in_frame += pothole_area_m2

            if show_boxes:
                cv2.rectangle(annotated_frame, (x1_box, y1_box), (x2_box, y2_box), box_color_bgr, 2)
                label_text = f"Area: {pothole_area_m2:.3f} m2"
                conf_text = f"Conf: {conf:.2f}"
                
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                max_text_w = max(label_w, conf_w)
                total_text_h = label_h + conf_h + 10 
                bg_y1 = y1_box - total_text_h 
                text_y_area = y1_box - conf_h - 5 
                text_y_conf = y1_box - 5 
                if bg_y1 < 0 : 
                    bg_y1 = y2_box + 5
                    text_y_area = y2_box + label_h + 5
                    text_y_conf = y2_box + label_h + conf_h + 10
                cv2.rectangle(annotated_frame, (x1_box, bg_y1), (x1_box + max_text_w + 10, bg_y1 + total_text_h), (50,50,50), -1)
                cv2.putText(annotated_frame, label_text, (x1_box + 5, text_y_area), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, conf_text, (x1_box + 5, text_y_conf), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
    return annotated_frame, pothole_details_current_frame, newly_detected_area_in_frame