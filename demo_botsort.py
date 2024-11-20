import cv2
from ultralytics import YOLO
import numpy as np
from utils import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

# 初始化模型和追踪器
video_path = r"C:\Users\11645\Desktop\Track\ultralytics-main\ultralytics\trackers\demo.mp4"
model_path = r"C:\Users\11645\Desktop\Track\ultralytics-main\ultralytics\trackers\best.pt"
model = YOLO(model_path)

cfg = IterableSimpleNamespace(**{
    'tracker_type': 'botsort',
    'track_high_thresh': 0.25,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.25,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'fuse_score': True,
    'gmc_method': 'sparseOptFlow',
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': False
})
tracker = BOTSORT(args=cfg, frame_rate=30)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 使用 YOLO 获取检测结果
    results = model.predict(frame)[0]

    # 提取 det 数据
    det = results.boxes.cpu().numpy() if results.boxes is not None else np.empty((0, 6))

    if len(det) == 0:
        continue

    # 调用 BOTSORT 的 update 方法
    tracks = tracker.update(det, img=frame)

    # 绘制追踪结果
    for track in tracks:
        x_min, y_min, x_max, y_max, track_id = track[:5]
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示结果
    cv2.imshow("BOTSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
