**README: YOLO Tracking System**

**1. Overview**

This project implements an object detection and tracking system using YOLO for object detection and BYTETracker for multi-object tracking. The script processes a given video file to detect objects in each frame and tracks their movements across frames.

**2. Requirements**

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

You can install the required dependencies using the following command:

```
pip install opencv-python ultralytics numpy
```

**3. Running the Script**

To run the tracking system, use the following parameters:

- `video_path`: Path to the input video file. For example: `demo.mp4`
- `model_path`: Path to the YOLO model weights file. For example: `best.pt`
- `tracker_cfg`: Dictionary containing configuration parameters for BYTETracker.

Example of running the tracking function:

```python
run_tracking(video_path, model_path, tracker_cfg)
```

**4. Configuration**

You can customize BYTETracker by modifying `tracker_cfg` parameters:

- `tracker_type`: The type of tracker used, set to `'bytetrack'`.
- `track_high_thresh`: High confidence threshold for track initialization.
- `track_low_thresh`: Low confidence threshold for potential tracks.
- `new_track_thresh`: Threshold to initialize a new track.
- `track_buffer`: The maximum number of frames a track can remain inactive.
- `match_thresh`: Threshold for matching detections with existing tracks.
- `fuse_score`: Boolean to enable score fusion.

**5. Usage**

- Once you run the script, it will open a window displaying the video with detected and tracked objects marked by rectangles and IDs.
- You can exit the tracking by pressing the "q" key.

**6. Notes**

- Ensure the model weights (`best.pt`) are compatible with the YOLO version being used.
- Adjust tracking parameters to suit your specific use case for optimal performance.

**7. License**

This project is provided under an MIT license. You can use, modify, and distribute it freely.

**二语版本：YOLO 跑道系统**

**1. 总览**

本项目实现了一个使用 YOLO 进行目标检测和 BYTETracker 进行多目标追踪的物体检测和追踪系统。该脚本对一个指定视频文件进行处理，在每个帧检测目标并追踪它们在帧上的运动。

**2. 软件环境要求**

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

可以使用以下命令安装必要的依赖：

```
pip install opencv-python ultralytics numpy
```

**3. 运行脚本**

使用以下参数运行追踪系统：

- `video_path`：输入视频文件的路径。例如：`demo.mp4`
- `model_path`：YOLO 模型权重文件的路径。例如：`best.pt`
- `tracker_cfg`：包含 BYTETracker 配置参数的字典。

例如：

```python
run_tracking(video_path, model_path, tracker_cfg)
```

**4. 配置信息**

可以通过修改 `tracker_cfg` 来自定义 BYTETracker ：

- `tracker_type`：追踪器类型，设为 `'bytetrack'`。
- `track_high_thresh`：追踪初始化的高信心闪值。
- `track_low_thresh`：未来追踪的低信心闪值。
- `new_track_thresh`：初始化新追踪的阀值。
- `track_buffer`：追踪在最大正常效的帧数。
- `match_thresh`：用于中本的追踪设置。
- `fuse_score`：是否开启平安展示。

