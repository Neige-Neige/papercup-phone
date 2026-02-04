"""测试摄像头"""
import cv2

print("测试 OpenCV 摄像头...")
print("=" * 40)

# 尝试不同的摄像头 ID
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[{i}] 可用! 分辨率: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"[{i}] 打开成功但无法读取画面")
        cap.release()
    else:
        print(f"[{i}] 无法打开")

print()
print("=" * 40)
print("如果上面没有找到可用摄像头，尝试 RealSense SDK...")

try:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) > 0:
        print(f"\n找到 {len(devices)} 个 RealSense 设备:")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev.get_info(rs.camera_info.name)}")
            print(f"      序列号: {dev.get_info(rs.camera_info.serial_number)}")
    else:
        print("\n没有找到 RealSense 设备")

except ImportError:
    print("\npyrealsense2 未安装")
    print("安装命令: pip install pyrealsense2")
except Exception as e:
    print(f"\nRealSense 检测出错: {e}")
