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
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.dataframe(mock_data, width='stretch')

# ================= TRANG 2: TRIỂN KHAI MÔ HÌNH =================
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
        
        # Xử lý lưu file tạm để cv2 có thể đọc (tương thích Streamlit Cloud)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()          
        os.fsync(tfile.fileno()) 
        tfile.close()            

        cap = cv2.VideoCapture(tfile.name)
        
        # Lấy FPS gốc của Video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0 or math.isnan(video_fps): 
            video_fps = 30.0
            
        required_frames_to_alarm = int(TIME_THRESHOLD * video_fps)
        closed_frames_count = 0
        
        stframe = st.empty() 
        status_text = st.empty() 
        audio_status = st.empty() 
        
        # Khởi tạo âm thanh cảnh báo
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

        start_time = time.time()
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                status_text.success("✅ Đã phát xong video!")
                break 
                
            # Resize khung hình để giảm tải tính toán
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
            
            # Hiển thị lên web
            stframe.image(rgb_frame, channels="RGB", width='stretch')
            
            # Đồng bộ hóa tốc độ phát thực tế (Real-time playback sync)
            processed_frames += 1
            if video_fps > 0:
                target_elapsed = processed_frames / video_fps 
                actual_elapsed = time.time() - start_time
                sleep_time = target_elapsed - actual_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Dọn dẹp tài nguyên
        if pygame_initialized:
            pygame.mixer.music.stop()
        cap.release()
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
            
            st.subheader("1. Tổng quan các chỉ số đo lường (Metrics)")
            st.markdown("Các chỉ số dưới đây phản ánh khả năng nhận diện thực tế của mô hình trên tổng số **{}** video kiểm thử:".format(total))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy (Độ chính xác)", f"{accuracy*100:.1f}%", help="Tỷ lệ dự đoán đúng trên tổng số video")
            col2.metric("Precision (Độ chuẩn xác)", f"{precision*100:.1f}%", help="Khả năng tránh báo động giả (False Positive)")
            col3.metric("Recall (Độ nhạy)", f"{recall*100:.1f}%", help="Khả năng không bỏ lọt ngủ gật (False Negative)")
            
            st.divider()
            st.subheader("2. Phân tích Trực quan (Data Visualization)")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Biểu đồ tỷ lệ các chỉ số hiệu năng**")
                metrics_df = pd.DataFrame({
                    'Chỉ số': ['Accuracy', 'Precision', 'Recall'],
                    'Giá trị (%)': [accuracy * 100, precision * 100, recall * 100]
                })
                
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Chỉ số', y='Giá trị (%)', data=metrics_df, palette=['#3498db', '#f1c40f', '#e74c3c'], ax=ax_bar)
                ax_bar.set_ylim(0, 100)
                ax_bar.set_ylabel('Phần trăm (%)')
                
                for p in ax_bar.patches:
                    ax_bar.annotate(format(p.get_height(), '.1f') + '%', 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha = 'center', va = 'center', 
                                    xytext = (0, 9), 
                                    textcoords = 'offset points', weight='bold')
                st.pyplot(fig_bar)
                st.caption("Biểu đồ so sánh 3 chỉ số quan trọng nhất của mô hình. Trong bài toán an toàn giao thông, Recall luôn được ưu tiên cao nhất.")

            with chart_col2:
                st.markdown("**Ma trận nhầm lẫn (Confusion Matrix Heatmap)**")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                cm_matrix = [[TN, FP], [FN, TP]]
                
                sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Dự đoán TỈNH', 'Dự đoán NGỦ'], 
                            yticklabels=['Thực tế TỈNH', 'Thực tế NGỦ'], 
                            annot_kws={"size": 16, "weight": "bold"}, ax=ax_cm)
                
                ax_cm.set_xlabel('Kết quả hệ thống (Predicted Label)')
                ax_cm.set_ylabel('Thực tế (True Label)')
                st.pyplot(fig_cm)
                st.caption(f"Bản đồ nhiệt thể hiện chi tiết phân bổ: TP={TP}, TN={TN}, FP={FP}, FN={FN}.")

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

    st.subheader("4. Giải thích hiện tượng Trade-off (Đánh đổi)")
    st.markdown("""
    Nhìn vào biểu đồ trên, có thể thấy hệ thống đang có xu hướng ưu tiên độ nhạy **(Recall cao hơn Precision)**. 
    * Trong lĩnh vực an toàn sinh mạng, việc hệ thống "Báo động giả" (False Positive - làm giảm Precision) chỉ gây ra sự phiền toái nhỏ cho tài xế. 
    * Tuy nhiên, nếu hệ thống "Bỏ lọt" (False Negative - làm giảm Recall) khi tài xế thực sự ngủ gật, hậu quả sẽ là một vụ tai nạn thảm khốc. 
    * $\\rightarrow$ Do đó, việc tinh chỉnh hệ thống nghiêng về Recall là một quyết định thiết kế có chủ đích.
    """)

    st.subheader("5. Hạn chế & Hướng phát triển")
    st.markdown("""
    * **Hạn chế:** Thông số `EAR_THRESHOLD` đang bị thiết lập cứng (Hardcoded). Điều này dẫn đến tỷ lệ sinh ra cảnh báo giả đối với những người có cấu trúc mắt bẩm sinh nhỏ (đặc trưng người Châu Á) hoặc khi người dùng nheo mắt do chói nắng.
    * **Hướng khắc phục:** Thu thập chuỗi dữ liệu EAR trong 5 phút lái xe đầu tiên lúc tài xế còn tỉnh táo, sau đó ứng dụng các thuật toán **Machine Learning có giám sát (SVM, KNN)** để tự động học và thiết lập "Ngưỡng EAR cá nhân hóa" (Personalized Baseline) cho từng khuôn mặt riêng biệt.
    """)

    st.divider()

    st.subheader("6. Nhật ký Video kiểm thử thủ công (App History)")
    if st.session_state['history_videos']:
        df_history = pd.DataFrame({
            'STT': range(1, len(st.session_state['history_videos']) + 1),
            'Tên file video': st.session_state['history_videos'],
            'Trạng thái kiểm tra': ['Hoàn tất luồng xử lý' for _ in st.session_state['history_videos']]
        })
        st.dataframe(df_history, width='stretch')
    else:
        st.info("Chưa có video nào được kiểm thử thủ công trên App. Hãy sang Trang 2 để tải video lên!")
