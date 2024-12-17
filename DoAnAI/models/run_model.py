import os
import gc
import csv
import datetime
from collections import defaultdict, deque

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.layers import (
    LSTM, Dense, TimeDistributed, Input, MultiHeadAttention,
    Masking, BatchNormalization, ReLU, Dropout
)
from tensorflow.keras.models import Model
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Constants and configurations
CONFIDENCE_THRESHOLD = 0.65
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Đường dẫn tới trọng số của mô hình mới vừa train
NEW_MODEL_WEIGHTS_PATH = r"C:\xampp\htdocs\DoAnAI\models\mobilenetv3_lstm_attention_multi_model.h5"

# Danh sách nhãn tương ứng với đầu ra của mô hình
action_labels = ['Read', 'Write', 'RaiseHand', 'Sleep']

# Thiết lập số bước thời gian (time steps) tối đa
MAX_SEQUENCE_LENGTH = 15

# Đường dẫn tới mô hình YOLOv11
YOLO_MODEL_PATH = "yolo11x.pt"


# Đường dẫn tới video đầu vào và đầu ra
INPUT_VIDEO_PATH = r"C:\xampp\htdocs\DoAnAI\Video\temp.mp4"
OUTPUT_VIDEO_PATH = r"C:\xampp\htdocs\DoAnAI\static\output1.mp4"

# Đường dẫn tới các tệp CSV và hình ảnh kết quả
STATISTICS_FILE = r"C:\xampp\htdocs\DoAnAI\static\action_statistics_filtered.csv"
TOTAL_SCORE_BAR_PLOT = r"C:\xampp\htdocs\DoAnAI\static\total_score_per_track_id_bar_plot.png"
ENGAGEMENT_PIE_CHART = r"C:\xampp\htdocs\DoAnAI\static\engagement_pie_chart.png"
ACTION_COUNTS_BAR_PLOT = r"C:\xampp\htdocs\DoAnAI\static\action_counts_per_track_id_bar_plot.png"
TOTAL_SCORES_WITH_ENGAGEMENT_CSV = r"C:\xampp\htdocs\DoAnAI\static\total_scores_with_engagement.csv"
ACTION_COUNTS_PER_TRACK_ID_CSV =r"C:\xampp\htdocs\DoAnAI\static\action_counts_per_track_id.csv"

# Tiền xử lý cho MobileNetV3
def preprocess_input_mobilenet_v3(x):
    return (x / 127.5) - 1

# Xây dựng mô hình với nhiều tầng trích xuất đặc trưng MobileNet
def build_mobilenet_model():
    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    base_model.trainable = False  # Đóng băng base model

    frame_input = Input(shape=(224, 224, 3))
    features = base_model(frame_input)

    x = BatchNormalization()(features)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    frame_encoder = Model(inputs=frame_input, outputs=x)

    # Xử lý chuỗi khung hình với nhiều tầng TimeDistributed
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, 224, 224, 3))
    encoded_sequence = TimeDistributed(frame_encoder)(sequence_input)

    # Thêm lớp Attention
    attention = MultiHeadAttention(num_heads=8, key_dim=64)(encoded_sequence, encoded_sequence)

    # Xử lý đặc trưng chuỗi với LSTM
    lstm_units = 128
    lstm_output = LSTM(lstm_units, return_sequences=False)(attention)
    x = Dropout(0.3)(lstm_output)

    # Dự đoán đầu ra
    output = Dense(len(action_labels), activation='softmax')(x)

    # Xây dựng và biên dịch mô hình
    mobilenet_model = Model(inputs=sequence_input, outputs=output)
    mobilenet_model.load_weights(NEW_MODEL_WEIGHTS_PATH)
    print("Loaded new model weights.")
    return mobilenet_model

# Tạo đối tượng VideoWriter
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

# Xây dựng và tải mô hình MobileNet
mobilenet_model = build_mobilenet_model()

# Mở video đầu vào và tạo VideoWriter
video_cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not video_cap.isOpened():
    print(f"Error: Cannot open video file {INPUT_VIDEO_PATH}")
    exit()
writer = create_video_writer(video_cap, OUTPUT_VIDEO_PATH)

# Tải mô hình YOLOv11
yolo_model = YOLO(YOLO_MODEL_PATH)
print("Loaded YOLOv11 model.")

# DeepSort tracker
tracker = DeepSort(max_age=100)

# Danh sách để lưu thông tin thống kê (ID, hành động, thời gian)
statistics = []
track_frame_sequences = {}
track_last_action = {}

fps = video_cap.get(cv2.CAP_PROP_FPS)  # Lấy FPS từ video

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    if not ret:
        break

    # Phát hiện đối tượng bằng YOLOv11
    detections = yolo_model(frame)[0]

    results = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    # Tracking
    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Cắt vùng hình ảnh
        cropped_frame = frame[max(0, ymin):min(frame.shape[0], ymax), max(0, xmin):min(frame.shape[1], xmax)]
        if cropped_frame.size == 0:
            continue

        resized_frame = cv2.resize(cropped_frame, (224, 224))
        x = img_to_array(resized_frame)
        x = preprocess_input_mobilenet_v3(x)

        if track_id not in track_frame_sequences:
            track_frame_sequences[track_id] = deque(maxlen=MAX_SEQUENCE_LENGTH)
        track_frame_sequences[track_id].append(x)

        if len(track_frame_sequences[track_id]) == MAX_SEQUENCE_LENGTH:
            sequence = np.array(track_frame_sequences[track_id])
            sequence = np.expand_dims(sequence, axis=0)

            predictions = mobilenet_model.predict(sequence)
            predicted_label = action_labels[np.argmax(predictions)]
            track_last_action[track_id] = predicted_label
            

            timestamp = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)
            statistics.append([track_id, predicted_label, timestamp])

            cv2.putText(frame, predicted_label, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        else:
            if track_id in track_last_action:
                cv2.putText(frame, track_last_action[track_id], (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    # Hiển thị FPS
    end = datetime.datetime.now()
    elapsed_time = (end - start).total_seconds()
    fps_display = f"FPS: {1 / elapsed_time:.2f}" if elapsed_time > 0 else "FPS: Inf"
    cv2.putText(frame, fps_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    writer.write(frame)

    # Giải phóng bộ nhớ
    gc.collect()

# Lọc các giá trị trùng `timestamp` và `track_id`, chỉ ghi khi `Action` thay đổi
unique_statistics = {}
for stat in statistics:
    track_id, action, timestamp = stat
    if track_id not in unique_statistics:
        unique_statistics[track_id] = {'last_action': None, 'records': []}

    # Kiểm tra sự thay đổi hành vi
    if action != unique_statistics[track_id]['last_action']:
        unique_statistics[track_id]['last_action'] = action
        unique_statistics[track_id]['records'].append([track_id, action, timestamp])

# Chuyển đổi lại thành danh sách các bản ghi đã lọc
filtered_statistics = []
for track_id, data in unique_statistics.items():
    filtered_statistics.extend(data['records'])

# Ghi thống kê vào file CSV
with open(STATISTICS_FILE, mode='w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(['Track ID', 'Action', 'Timestamp (Seconds)'])
    for stat in filtered_statistics:
        writer_csv.writerow(stat)

print(f"Filtered statistics have been written to '{STATISTICS_FILE}'.")

video_cap.release()
writer.release()
print("Processing completed and statistics saved.")

# Phân tích thống kê và tạo biểu đồ
def analyze_statistics(statistics_file):
    # Các nhãn hành động cần phân biệt
    positive_actions = ['RaiseHand', 'Read', 'Write']
    negative_actions = ['Sleep']

    # Số lần thực hiện hành động và điểm của từng ID
    action_counts = defaultdict(int)
    id_scores = defaultdict(int)

    # Đọc dữ liệu từ tệp CSV
    with open(statistics_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua tiêu đề
        for row in reader:
            track_id = int(row[0])
            action = row[1]
            timestamp = int(row[2])

            # Đếm số lần thực hiện hành động
            action_counts[(track_id, action)] += 1

            # Tính điểm cho từng ID và hành động
            if action in positive_actions:
                if action == 'RaiseHand':
                    action_score = 3
                else:
                    action_score = 1
            elif action in negative_actions:
                action_score = -10
            else:
                action_score = 0

            # Cập nhật điểm cho track_id
            id_scores[track_id] += action_score

    # In ra số lần thực hiện hành động của từng ID
    print("Number of times each action was performed by each ID:")
    for (track_id, action), count in action_counts.items():
        print(f"Track ID: {track_id}, Action: {action}, Count: {count}")

    # In ra tổng điểm cho từng ID
    print("\nTotal score for each ID:")
    for track_id, score in id_scores.items():
        print(f"Track ID: {track_id}, Total Score: {score}")

    # Vẽ biểu đồ cột cho điểm tổng mỗi ID
    track_ids = list(id_scores.keys())
    total_scores = list(id_scores.values())

    plt.figure(figsize=(12, 6))
    plt.bar(track_ids, total_scores, color='skyblue', edgecolor='black')
    plt.title('Total Score per Track ID')
    plt.xlabel('Track ID')
    plt.ylabel('Total Score')
    plt.savefig(TOTAL_SCORE_BAR_PLOT)  # Lưu biểu đồ
    plt.close()

    # Lưu kết quả vào tệp CSV với cột Engagement
    with open(TOTAL_SCORES_WITH_ENGAGEMENT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Track ID', 'Total Score', 'Engagement'])  # Thêm cột Engagement
        # Thống kê số lượng trạng thái Engagement, Neutral và Not Engagement
        engagement_count = {'Engagement': 0, 'Neutral': 0, 'Not Engagement': 0}

        # Đếm số lượng các trạng thái
        for track_id, score in id_scores.items():
            if score > 0:
                engagement_count['Engagement'] += 1
                engagement = 'Engagement'
            elif score == 0:
                engagement_count['Neutral'] += 1
                engagement = 'Neutral'
            else:
                engagement_count['Not Engagement'] += 1
                engagement = 'Not Engagement'
            # Ghi thông tin vào file CSV
            writer.writerow([track_id, score, engagement])

        # Tính toán phần trăm của mỗi trạng thái
        total_ids = len(id_scores)
        engagement_percent = (engagement_count['Engagement'] / total_ids) * 100 if total_ids else 0
        neutral_percent = (engagement_count['Neutral'] / total_ids) * 100 if total_ids else 0
        not_engagement_percent = (engagement_count['Not Engagement'] / total_ids) * 100 if total_ids else 0

        # Vẽ biểu đồ tròn
        labels = ['Engagement', 'Neutral', 'Not Engagement']
        sizes = [engagement_percent, neutral_percent, not_engagement_percent]
        colors = ['#66b3ff', '#99cc99', '#ff6666']
        explode = (0.1, 0, 0)  # Nổi bật phần Engagement

        plt.figure(figsize=(8, 8))
        plt.pie(
            sizes,
            colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',  # Chỉ hiển thị phần trăm > 0
            startangle=140,
            wedgeprops={'edgecolor': 'black'}
        )
        plt.title('Engagement Status Distribution (%)')

        # Thêm chú thích
        plt.legend(labels, title="Legend", loc="upper right")

        # Lưu biểu đồ tròn dưới dạng hình ảnh (PNG)
        plt.savefig(ENGAGEMENT_PIE_CHART)  # Lưu biểu đồ tròn
        plt.close()

    print(f"Results saved to image and CSV file:\n - Bar Plot saved as '{TOTAL_SCORE_BAR_PLOT}'\n - Total Scores with Engagement saved to '{TOTAL_SCORES_WITH_ENGAGEMENT_CSV}'")

# Phân tích các hành động tích cực và tiêu cực
def analyze_positive_negative_actions(statistics_file):
    positive_actions = ['RaiseHand', 'Read', 'Write']
    negative_actions = ['Sleep']

    positive_action_counts = defaultdict(int)
    negative_action_counts = defaultdict(int)

    # Đọc dữ liệu từ tệp CSV
    with open(statistics_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            track_id = int(row[0])
            action = row[1]
            timestamp = int(row[2])

            if action in positive_actions:
                positive_action_counts[track_id] += 1
            elif action in negative_actions:
                negative_action_counts[track_id] += 1

    # In ra số lượng hành động tích cực và tiêu cực của từng Track ID
    print("Number of positive and negative actions for each Track ID:")
    for track_id in sorted(set(positive_action_counts.keys()).union(negative_action_counts.keys())):
        positive_count = positive_action_counts.get(track_id, 0)
        negative_count = negative_action_counts.get(track_id, 0)
        print(f"Track ID: {track_id}, Positive Actions: {positive_count}, Negative Actions: {negative_count}")

    # Vẽ biểu đồ cột cho số lượng hành động tích cực và tiêu cực
    track_ids = sorted(set(positive_action_counts.keys()).union(negative_action_counts.keys()))
    positive_counts = [positive_action_counts.get(track_id, 0) for track_id in track_ids]
    negative_counts = [negative_action_counts.get(track_id, 0) for track_id in track_ids]

    x = np.arange(len(track_ids))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, positive_counts, width, label='Positive Actions', color='skyblue', edgecolor='black')
    plt.bar(x + width, negative_counts, width, label='Negative Actions', color='salmon', edgecolor='black')

    plt.title('Number of Positive and Negative Actions per Track ID')
    plt.xlabel('Track ID')
    plt.ylabel('Number of Actions')
    plt.xticks(x + width / 2, track_ids)
    plt.legend()

    plt.savefig(ACTION_COUNTS_BAR_PLOT)
    plt.close()

    # Lưu kết quả vào tệp CSV
    with open(ACTION_COUNTS_PER_TRACK_ID_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Track ID', 'Positive Actions', 'Negative Actions'])

        for track_id in track_ids:
            positive_count = positive_action_counts.get(track_id, 0)
            negative_count = negative_action_counts.get(track_id, 0)
            writer.writerow([track_id, positive_count, negative_count])

    print(f"Results saved to image and CSV file:\n - Action Counts Bar Plot saved as '{ACTION_COUNTS_BAR_PLOT}'\n - Action Counts saved to '{ACTION_COUNTS_PER_TRACK_ID_CSV}'")

# Thực thi các phân tích
analyze_statistics(STATISTICS_FILE)
analyze_positive_negative_actions(STATISTICS_FILE)

print(f"All results have been saved successfully.")
