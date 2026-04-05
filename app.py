import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import math
import tempfile
import time
import os
import pygame 
import json

# ================= CẤU HÌNH TRANG WEB =================
st.set_page_config(page_title="Hệ thống Cảnh báo Ngủ gật", page_icon="🚗", layout="wide")

# Khởi tạo bộ nhớ tạm để lưu danh sách video đã upload
if 'history_videos' not in st.session_state:
    st.session_state['history_videos'] = []

# ================= KHỞI TẠO CACHE CHO MÔ HÌNH (MLOps) =================
@st.cache_resource
def load_face_mesh():
    """Hàm load mô hình MediaPipe được cache lại để tối ưu tốc độ"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True), mp.solutions.drawing_utils, mp.solutions.drawing_styles

face_mesh, mp_drawing, mp_drawing_styles = load_face_mesh()

# ================= HẰNG SỐ & HÀM XỬ LÝ TOÁN HỌC =================
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h = euclidean_distance(p1, p4)
    return (v1 + v2) / (2.0 * h)

# ================= MỤC LỤC ĐIỀU HƯỚNG =================
st.sidebar.title("📌 Danh mục")
page = st.sidebar.radio("Chọn trang chức năng:", 
                        ["1. Giới thiệu & Khám phá dữ liệu", 
                         "2. Triển khai mô hình", 
                         "3. Đánh giá & Hiệu năng"])

# ================= TRANG 1: GIỚI THIỆU & EDA =================
if page == "1. Giới thiệu & Khám phá dữ liệu":
    st.title("🚗 HỆ THỐNG NHẬN DIỆN VÀ CẢNH BÁO TÀI XẾ NGỦ GẬT")
    st.markdown("""
    **Thông tin sinh viên:**
    * **Họ và tên:** Nguyễn Trọng Nghĩa
    * **MSSV:** 22T1020256
    
    ### 🎯 Mục tiêu & Giá trị thực tiễn
    Hệ thống giúp phát hiện sớm dấu hiệu giảm tỉnh táo qua luồng video bằng cách tính toán Tỷ lệ khung mắt (EAR). 
    Dự án ứng dụng mô hình Deep Learning **MediaPipe Face Mesh** của Google để bắt 468 điểm landmark trên mặt kết hợp với thuật toán logic chuỗi thời gian.
    """)
    st.divider()
    
    st.subheader("🔍 Mẫu dữ liệu thô trích xuất từ Video (Raw Data)")
    st.markdown("Dưới đây là mô phỏng cấu trúc dữ liệu thô (chuỗi số EAR) được hệ thống trích xuất theo từng khung hình (frame) để phục vụ cho logic ra quyết định và mở rộng huấn luyện Học máy (MLOps) trong tương lai:")
    
    mock_data = pd.DataFrame({
        'Khung_hình (Frame)': [1, 2, 3, 4, 5, 6],
        'EAR_Mắt_Trái': [0.320, 0.315, 0.150, 0.120, 0.110, 0.300],
        'EAR_Mắt_Phải': [0.310, 0.320, 0.140, 0.115, 0.105, 0.290],
        'EAR_Trung_bình': [0.315, 0.317, 0.145, 0.117, 0.107, 0.295],
        'Đánh_giá_Tức_thời': ['Mở mắt', 'Mở mắt', 'Nhắm mắt', 'Nhắm mắt', 'Nhắm mắt', 'Mở mắt']
    })
    st.dataframe(mock_data, width='tretch')

# ================= TRANG 2: TRIỂN KHAI MÔ HÌNH (ĐÃ SỬA ĐỂ CHẠY REAL-TIME TRÊN STREAMLIT CLOUD) =================
elif page == "2. Triển khai mô hình":
    st.title("⚙️ Triển khai nhận diện qua Video")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Cài đặt thông số")
    EAR_THRESHOLD = st.sidebar.slider("Ngưỡng EAR (Threshold)", 0.15, 0.35, 0.21, 0.01)
    TIME_THRESHOLD = st.sidebar.slider("Thời gian nhắm mắt (Giây)", 0.5, 3.0, 1.5, 0.1)
    SHOW_MESH = st.sidebar.checkbox("👁️ Hiện lưới điểm mặt", value=False)

    uploaded_file = st.file_uploader("Tải video kiểm thử (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state['history_videos']:
            st.session_state['history_videos'].append(uploaded_file.name)

        st.info(f"Đang xử lý: {uploaded_file.name}")
        
        # ====================== SỬA LỖI THƯỜNG GẶP TRÊN STREAMLIT CLOUD ======================
        # 1. Tạo temp file + flush + fsync để đảm bảo file đã ghi đầy đủ trước khi cv2 đọc
        # 2. Thêm delay để video chạy đúng tốc độ gốc (real-time playback)
        # 3. Tối ưu hơn để tránh lag UI trên Cloud
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()           # Quan trọng: ép dữ liệu xuống đĩa
        os.fsync(tfile.fileno()) # Đảm bảo file hoàn chỉnh trước khi cv2 mở
        tfile.close()            # Đóng file để cv2 có thể đọc

        cap = cv2.VideoCapture(tfile.name)
        
        # --- LẤY FPS GỐC CỦA VIDEO ---
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0 or math.isnan(video_fps): 
            video_fps = 30.0
            
        required_frames_to_alarm = int(TIME_THRESHOLD * video_fps)
        closed_frames_count = 0
        
        stframe = st.empty() 
        status_text = st.empty() 
        audio_status = st.empty() 
        
        # --- KHỞI TẠO PYGAME ---
        alarm_sound_path = "alarm.mp3" 
        pygame_initialized = False
        
        if not os.path.exists(alarm_sound_path):
            audio_status.error(f"❌ KHÔNG TÌM THẤY FILE: '{alarm_sound_path}'. Vui lòng copy file này vào cùng thư mục với app.py!")
        else:
            try:
                pygame.mixer.init()
                pygame_initialized = True
            except Exception as e:
                audio_status.warning(f"⚠️ Không thể khởi tạo âm thanh (Bình thường nếu chạy trên Cloud): {e}")

        # ====================== REAL-TIME PLAYBACK LOGIC ======================
        start_time = time.time()
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                status_text.success("✅ Đã phát xong video!")
                break 
                
            # Tối ưu kích thước (giảm lag trên Cloud)
            h_ori, w_ori, _ = frame.shape
            new_w = 640
            new_h = int((new_w / w_ori) * h_ori)
            frame = cv2.resize(frame, (new_w, new_h))

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if SHOW_MESH:
                        mp_drawing.draw_landmarks(
                            image=rgb_frame, 
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
                    right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    color = (0, 255, 0)
                    
                    if avg_ear < EAR_THRESHOLD:
                        closed_frames_count += 1
                        elapsed_video_time = closed_frames_count / video_fps
                        
                        if closed_frames_count >= required_frames_to_alarm:
                            color = (255, 0, 0)
                            status_text.error(f"🚨 CẢNH BÁO: NGỦ GẬT! (EAR: {avg_ear:.2f} - Đã nhắm mắt {elapsed_video_time:.1f}s)")
                            
                            if pygame_initialized and not pygame.mixer.music.get_busy():
                                try:
                                    pygame.mixer.music.load(alarm_sound_path)
                                    pygame.mixer.music.play(-1)
                                except:
                                    pass
                    else:
                        closed_frames_count = 0
                        status_text.success(f"✅ ĐANG TỈNH TÁO (EAR: {avg_ear:.2f})")
                        
                        if pygame_initialized and pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                    
                    cv2.putText(rgb_frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # HIỂN THỊ FRAME (đã được fix để chạy mượt trên Cloud)
            stframe.image(rgb_frame, channels="RGB", width='tretch')
            
            # ====================== ĐIỀU CHỈNH TỐC ĐỘ REAL-TIME ======================
            processed_frames += 1
            if video_fps > 0:
                target_elapsed = processed_frames / video_fps          # Thời gian lý thuyết theo FPS gốc
                actual_elapsed = time.time() - start_time
                sleep_time = target_elapsed - actual_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)   # Giữ video chạy đúng tốc độ gốc

        # Dọn dẹp
        if pygame_initialized:
            pygame.mixer.music.stop()
        cap.release()
        
        # Xóa file tạm (tốt cho Cloud)
        try:
            os.unlink(tfile.name)
        except:
            pass

# ================= TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG =================
elif page == "3. Đánh giá & Hiệu năng":
    st.title("📈 Báo cáo Đánh giá & Tiêu chí Hiệu năng")
    
    st.info("Hệ thống được đánh giá qua Kiểm thử tự động (Automated Batch Testing) trên tập dữ liệu video thực tế nhằm đối chiếu kết quả dự đoán với nhãn gốc (Ground Truth).")

    if os.path.exists("evaluation_metrics.json"):
        with open("evaluation_metrics.json", "r") as f:
            metrics_data = json.load(f)
            
        TP = metrics_data["TP"]
        TN = metrics_data["TN"]
        FP = metrics_data["FP"]
        FN = metrics_data["FN"]
        
        total = TP + TN + FP + FN
        
        if total > 0:
            accuracy = (TP + TN) / total
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            st.subheader("1. Các chỉ số đo lường (Metrics)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy (Độ chính xác)", f"{accuracy*100:.1f}%", help="Tỷ lệ dự đoán đúng trên tổng số video")
            col2.metric("Precision (Độ chuẩn xác)", f"{precision*100:.1f}%", help="Khả năng tránh báo động giả (False Positive)")
            col3.metric("Recall (Độ nhạy)", f"{recall*100:.1f}%", help="Khả năng không bỏ lọt ngủ gật (False Negative)")
            
            st.subheader("2. Công cụ trực quan (Charts) - Ma trận nhầm lẫn")
            col_chart1, col_chart2 = st.columns([1.2, 1])
            
            with col_chart1:
                conf_matrix = pd.DataFrame(
                    [[TN, FP], [FN, TP]], 
                    index=['Thực tế: Tỉnh táo (0)', 'Thực tế: Ngủ gật (1)'],
                    columns=['Dự đoán: Tỉnh táo (0)', 'Dự đoán: Ngủ gật (1)']
                )
                st.table(conf_matrix)
                st.caption(f"**Tổng số mẫu kiểm thử:** {total} video. Bảng thể hiện sự tương quan giữa thực tế và dự đoán của hệ thống.")

            with col_chart2:
                chart_df = pd.DataFrame({
                    "Phân loại": ["Dự đoán ĐÚNG (TP+TN)", "Dự đoán SAI (FP+FN)"],
                    "Số lượng video": [TP + TN, FP + FN]
                }).set_index("Phân loại")
                st.bar_chart(chart_df, color="#2ecc71")
        else:
            st.warning("Dữ liệu đánh giá trống. Hãy thêm video vào thư mục test_dataset và chạy lại auto_test.py!")
    else:
        st.warning("⚠️ Chưa tìm thấy file `evaluation_metrics.json`. Vui lòng chạy file `auto_test.py` trên Terminal để hệ thống tự động sinh dữ liệu biểu đồ!")

    st.divider()

    st.subheader("3. Phương pháp & Logic Đánh giá")
    st.markdown("""
    Quyết định cảnh báo của hệ thống không dựa trên cảm tính, mà được đánh giá khắt khe qua 2 rào cản đồng thời:
    * **Đánh giá Không gian (Ngưỡng EAR):** Sử dụng khoảng cách hình học Euclidean để đo lường tỷ lệ mở của mắt. Khi EAR tụt xuống dưới ngưỡng, ghi nhận mắt đang nhắm.
    * **Đánh giá Thời gian (Time Threshold):** Loại trừ các chớp mắt sinh lý bình thường bằng cách bắt buộc trạng thái nhắm mắt phải duy trì liên tục vượt qua ngưỡng quy định (ví dụ: `> 1.5s`).
    """)

    st.subheader("4. Đánh giá Ưu điểm và Hạn chế")
    st.markdown("""
    **🔥 Điểm mạnh (Dựa trên thực nghiệm):**
    * **Hoạt động Thời gian thực (Real-time):** Tốc độ xử lý cao, độ trễ cực thấp do không load các mô hình phân loại ảnh nặng nề.
    * **Tối ưu tài nguyên:** Hệ thống chạy mượt mà trên thiết bị không có GPU.
    * **Kháng nhiễu ánh sáng:** Đánh giá dựa trên tọa độ điểm ảnh (landmarks), ít bị ảnh hưởng bởi môi trường thiếu sáng hơn phương pháp phân tích pixel truyền thống.

    **⚠️ Hạn chế & Hướng phát triển:**
    * Thông số `EAR_THRESHOLD` đang bị cố định (Cứng). Dựa trên biểu đồ ma trận nhầm lẫn phía trên, hệ thống vẫn có tỷ lệ sinh ra *Báo động giả (False Positives)* nếu tài xế có đặc điểm mắt nhỏ hoặc góc đặt camera không thuận lợi.
    * **Hướng khắc phục:** Trong tương lai, cần tích hợp thuật toán Học máy có giám sát (như SVM hoặc KNN) sử dụng file CSV trích xuất để hệ thống có khả năng tự động học và tìm ra ngưỡng EAR phù hợp cho từng khuôn mặt riêng biệt (Personalization).
    """)

    st.divider()

    st.subheader("5. Nhật ký Video kiểm thử thủ công (App History)")
    if st.session_state['history_videos']:
        df_history = pd.DataFrame({
            'STT': range(1, len(st.session_state['history_videos']) + 1),
            'Tên file video': st.session_state['history_videos'],
            'Trạng thái kiểm tra': ['Hoàn tất luồng xử lý' for _ in st.session_state['history_videos']]
        })
        st.dataframe(df_history, width='tretch')
    else:
        st.info("Chưa có video nào được kiểm thử thủ công trên App. Hãy sang Trang 2 để tải video lên!")
