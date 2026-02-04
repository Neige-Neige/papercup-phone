"""
MCP Smart Listener Server
结合 YAMNet (声音事件分类) + Whisper (语音转录) + 摄像头 的智能感知服务

功能:
1. listen - 实时录音并智能分析
2. classify_sound - 仅识别声音类型
3. transcribe_speech - 仅转录语音
4. analyze_file - 分析音频文件
5. start_monitor - 启动环境常驻监听
6. stop_monitor - 停止监听
7. get_monitor_events - 获取监听到的事件
8. capture_camera - 拍照
9. auto_monitor_loop - 多模态监控（声音+图像）
"""

import asyncio
import logging
import tempfile
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import base64

# Whisper
import whisper

# Camera
try:
    import cv2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("注意: opencv 未安装，摄像头功能不可用。安装: pip install opencv-python")

# MCP
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-listener")

# ============== 模型加载 ==============

print("正在加载 YAMNet 模型...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 加载 YAMNet 类别名称
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
yamnet_classes = list(pd.read_csv(class_map_path)['display_name'])

print("正在加载 Whisper 模型...")
# 可选: tiny, base, small, medium, large
# tiny/base 速度快，large 准确率高
whisper_model = whisper.load_model("base")

print("模型加载完成!")

# 语音相关的 YAMNet 类别索引
SPEECH_CLASSES = [
    "Speech", "Narration, monologue", "Conversation",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Whispering", "Shout", "Yell",
    "Singing", "Chant"
]



# ============== 摄像头功能 ==============

# 照片保存目录
PHOTO_CACHE_DIR = Path(__file__).parent / "photo_cache"
PHOTO_CACHE_DIR.mkdir(exist_ok=True)

def capture_camera(camera_id: int = 0, warmup_frames: int = 10) -> dict:
    """拍照（带预热帧让摄像头自动调整曝光）"""
    if not CAMERA_AVAILABLE:
        return {"error": "opencv 未安装，请运行: pip install opencv-python"}

    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {"error": f"无法打开摄像头 {camera_id}"}

        # 预热：读取几帧让摄像头自动调整曝光
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.05)  # 稍微等一下

        # 正式拍照
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"error": "无法读取摄像头画面"}

        # 保存到本地
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = PHOTO_CACHE_DIR / filename
        cv2.imwrite(str(filepath), frame)
        logger.info(f"照片已保存: {filepath}")

        # 转为 JPEG 并编码为 base64
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf).decode()

        return {
            "success": True,
            "image_b64": image_b64,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "saved_path": str(filepath)
        }
    except Exception as e:
        return {"error": str(e)}

# ============== 环境监听状态 ==============

class EnvironmentMonitor:
    """环境常驻监听器"""

    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.events = deque(maxlen=100)  # 最多保存100个事件
        self.watch_classes = []  # 要关注的声音类别
        self.threshold = 0.3  # 检测阈值
        self.interval = 2.0  # 监听间隔(秒)
        self.transcribe_speech = True  # 是否转录语音

    def start(self, watch_classes: list[str] = None, threshold: float = 0.3,
              interval: float = 2.0, transcribe: bool = True):
        """启动监听"""
        if self.is_running:
            return False

        self.watch_classes = watch_classes or []
        self.threshold = threshold
        self.interval = interval
        self.transcribe_speech = transcribe
        self.is_running = True
        self.events.clear()

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """停止监听"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        return True

    def _monitor_loop(self):
        """监听循环"""
        logger.info("环境监听已启动")

        while self.is_running:
            try:
                # 录制一小段音频
                audio = record_audio(self.interval)

                # 分类
                top_classes, speech_score = classify_with_yamnet(audio)

                # 检查是否有关注的声音
                detected = []
                for class_name, score in top_classes:
                    # 如果没有指定关注类别，所有高于阈值的都记录
                    if not self.watch_classes:
                        if score >= self.threshold:
                            detected.append((class_name, score))
                    # 如果指定了关注类别，只记录匹配的
                    else:
                        for watch in self.watch_classes:
                            if watch.lower() in class_name.lower() and score >= self.threshold:
                                detected.append((class_name, score))
                                break

                # 如果检测到关注的声音，记录事件
                if detected:
                    event = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "sounds": detected,
                        "speech_score": speech_score,
                        "transcription": None
                    }

                    # 如果检测到语音且需要转录
                    if self.transcribe_speech and speech_score >= 0.3:
                        try:
                            result = transcribe_with_whisper(audio)
                            if result["text"].strip():
                                event["transcription"] = result
                        except Exception as e:
                            logger.error(f"转录失败: {e}")

                    self.events.append(event)
                    logger.info(f"检测到事件: {detected}")

            except Exception as e:
                logger.error(f"监听循环错误: {e}")

            # 短暂休息避免 CPU 过载
            time.sleep(0.1)

        logger.info("环境监听已停止")

    def get_events(self, clear: bool = False) -> list[dict]:
        """获取事件列表"""
        events = list(self.events)
        if clear:
            self.events.clear()
        return events

    def get_status(self) -> dict:
        """获取监听状态"""
        return {
            "is_running": self.is_running,
            "watch_classes": self.watch_classes,
            "threshold": self.threshold,
            "interval": self.interval,
            "event_count": len(self.events)
        }

# 全局监听器实例
monitor = EnvironmentMonitor()

# ============== 工具函数 ==============

def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """录制音频"""
    logger.info(f"开始录音 {duration} 秒...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    logger.info("录音完成")
    return audio.flatten()


def classify_with_yamnet(audio: np.ndarray) -> tuple[list[tuple[str, float]], float]:
    """
    使用 YAMNet 分类声音
    返回: (top_classes, speech_score)
    """
    scores, embeddings, spectrogram = yamnet_model(audio)
    mean_scores = scores.numpy().mean(axis=0)

    # 获取 top 10 分类
    top_indices = np.argsort(mean_scores)[-10:][::-1]
    top_classes = [
        (yamnet_classes[idx], float(mean_scores[idx]))
        for idx in top_indices
        if mean_scores[idx] > 0.02
    ]

    # 计算语音得分 - 累加所有语音相关类别的分数
    speech_score = 0.0
    for class_name in SPEECH_CLASSES:
        if class_name in yamnet_classes:
            idx = yamnet_classes.index(class_name)
            speech_score += mean_scores[idx]

    # 也检查 top_classes 里是否有语音相关的
    for class_name, score in top_classes:
        for speech_class in SPEECH_CLASSES:
            if speech_class.lower() in class_name.lower():
                speech_score = max(speech_score, score + 0.1)  # 加一点权重
                break

    # 限制在 0-1 之间
    speech_score = min(speech_score, 1.0)

    logger.info(f"语音得分: {speech_score:.2f}, Top: {top_classes[:3]}")

    return top_classes, speech_score


def transcribe_with_whisper(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    使用 Whisper 转录语音
    """
    # Whisper 需要 float32 格式，16kHz
    # 保存为临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        temp_path = f.name

    try:
        result = whisper_model.transcribe(
            temp_path,
            language=None,  # 自动检测语言
            task="transcribe"
        )
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown")
        }
    finally:
        os.unlink(temp_path)


def format_classifications(classes: list[tuple[str, float]]) -> str:
    """格式化分类结果"""
    lines = []
    for name, score in classes[:7]:
        bar = "█" * int(score * 20)
        lines.append(f"  {name}: {score:.1%} {bar}")
    return "\n".join(lines)


# ============== MCP Server ==============

server = Server("smart-listener")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="listen",
            description="智能听音：录制音频，自动识别声音类型。如果检测到语音则同时转录内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "录音时长(秒)，默认 5 秒",
                        "default": 5
                    },
                    "speech_threshold": {
                        "type": "number",
                        "description": "语音检测阈值 (0-1)，默认 0.15",
                        "default": 0.15
                    }
                }
            }
        ),
        Tool(
            name="classify_sound",
            description="仅识别声音类型（使用 YAMNet），不转录语音内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "录音时长(秒)",
                        "default": 3
                    }
                }
            }
        ),
        Tool(
            name="transcribe_speech",
            description="仅转录语音内容（使用 Whisper），不分类声音",
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "录音时长(秒)",
                        "default": 5
                    }
                }
            }
        ),
        Tool(
            name="analyze_file",
            description="分析音频文件，同时进行声音分类和语音转录",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "音频文件路径 (支持 wav, mp3, flac 等)"
                    }
                },
                "required": ["filepath"]
            }
        ),
        Tool(
            name="list_audio_devices",
            description="列出可用的音频输入设备",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="start_monitor",
            description="启动环境常驻监听。持续监听环境声音，检测到指定类型的声音时记录事件。",
            inputSchema={
                "type": "object",
                "properties": {
                    "watch_classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要关注的声音类别列表，如 ['dog', 'door', 'speech']。留空则监听所有声音。"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "检测阈值 (0-1)，默认 0.3",
                        "default": 0.3
                    },
                    "interval": {
                        "type": "number",
                        "description": "监听间隔(秒)，默认 2 秒",
                        "default": 2.0
                    },
                    "transcribe": {
                        "type": "boolean",
                        "description": "是否转录检测到的语音，默认 true",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="stop_monitor",
            description="停止环境常驻监听",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_monitor_events",
            description="获取环境监听检测到的事件列表",
            inputSchema={
                "type": "object",
                "properties": {
                    "clear": {
                        "type": "boolean",
                        "description": "获取后是否清空事件列表，默认 false",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="get_monitor_status",
            description="获取环境监听器的当前状态",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="capture_camera",
            description="拍一张照片让 Claude 分析（表情、动作、物体、场景）📷",
            inputSchema={
                "type": "object",
                "properties": {
                    "camera_id": {
                        "type": "number",
                        "description": "摄像头 ID，默认 0（内置摄像头）",
                        "default": 0
                    }
                }
            }
        ),
        Tool(
            name="auto_monitor_loop",
            description="""【多模态监控】听到声音时自动拍照！🎤📷

检测到声音 → 自动拍照 → 返回声音类型 + 画面
例如：听到咳嗽 + 看到揉眼睛 → "看起来累了，休息一下？"

响应后记得继续调用这个工具来保持监控！""",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_wait_seconds": {
                        "type": "number",
                        "description": "最大等待时间（秒），默认 60",
                        "default": 60
                    },
                    "sensitivity": {
                        "type": "number",
                        "description": "检测灵敏度 (0-1)，越低越敏感，默认 0.2",
                        "default": 0.2
                    },
                    "camera_id": {
                        "type": "number",
                        "description": "摄像头 ID，默认 0",
                        "default": 0
                    }
                }
            }
        ),
        Tool(
            name="listen_and_respond",
            description="""等待并监听有意义的声音，返回声音类型和响应建议。

忽略静音和噪音，只在检测到有意义的声音时返回。
例如：咳嗽、笑声、敲门、狗叫等。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_wait_seconds": {
                        "type": "number",
                        "description": "最大等待时间（秒），默认 30",
                        "default": 30
                    },
                    "sensitivity": {
                        "type": "number",
                        "description": "检测灵敏度 (0-1)，默认 0.25",
                        "default": 0.25
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""

    if name == "listen":
        return await smart_listen(
            duration=arguments.get("duration", 5),
            speech_threshold=arguments.get("speech_threshold", 0.3)
        )

    elif name == "classify_sound":
        return await classify_sound_only(
            duration=arguments.get("duration", 3)
        )

    elif name == "transcribe_speech":
        return await transcribe_speech_only(
            duration=arguments.get("duration", 5)
        )

    elif name == "analyze_file":
        return await analyze_audio_file(
            filepath=arguments["filepath"]
        )

    elif name == "list_audio_devices":
        return await list_audio_devices()

    elif name == "start_monitor":
        return await start_environment_monitor(
            watch_classes=arguments.get("watch_classes", []),
            threshold=arguments.get("threshold", 0.3),
            interval=arguments.get("interval", 2.0),
            transcribe=arguments.get("transcribe", True)
        )

    elif name == "stop_monitor":
        return await stop_environment_monitor()

    elif name == "get_monitor_events":
        return await get_monitor_events(
            clear=arguments.get("clear", False)
        )

    elif name == "get_monitor_status":
        return await get_monitor_status()

    elif name == "capture_camera":
        return await capture_camera_tool(
            camera_id=int(arguments.get("camera_id", 0))
        )

    elif name == "auto_monitor_loop":
        return await auto_monitor_loop(
            max_wait_seconds=arguments.get("max_wait_seconds", 60),
            sensitivity=arguments.get("sensitivity", 0.2),
            camera_id=int(arguments.get("camera_id", 0))
        )

    elif name == "listen_and_respond":
        return await listen_and_respond(
            max_wait_seconds=arguments.get("max_wait_seconds", 30),
            sensitivity=arguments.get("sensitivity", 0.25)
        )

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def smart_listen(duration: float, speech_threshold: float) -> list[TextContent]:
    """智能听音：结合 YAMNet + Whisper"""

    # 在线程池中运行录音（避免阻塞）
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, record_audio, duration)

    # YAMNet 分类
    top_classes, speech_score = await loop.run_in_executor(
        None, classify_with_yamnet, audio
    )

    result_lines = ["🎧 智能听音分析结果", "=" * 30, ""]

    # 声音分类结果
    result_lines.append("📊 检测到的声音类型:")
    result_lines.append(format_classifications(top_classes))
    result_lines.append("")
    result_lines.append(f"🗣️ 语音检测得分: {speech_score:.1%} (阈值: {speech_threshold:.1%})")
    result_lines.append("")

    # 如果检测到语音，进行转录
    if speech_score >= speech_threshold:
        result_lines.append("✅ 检测到语音，正在转录...")

        transcription = await loop.run_in_executor(
            None, transcribe_with_whisper, audio
        )

        result_lines.append("")
        result_lines.append(f"📝 转录结果 [{transcription['language']}]:")
        result_lines.append(f"   \"{transcription['text']}\"")
    else:
        result_lines.append("ℹ️ 未检测到明显语音，跳过转录")

    return [TextContent(type="text", text="\n".join(result_lines))]


async def classify_sound_only(duration: float) -> list[TextContent]:
    """仅进行声音分类"""

    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, record_audio, duration)

    top_classes, speech_score = await loop.run_in_executor(
        None, classify_with_yamnet, audio
    )

    result_lines = ["🔊 声音分类结果 (YAMNet)", "=" * 30, ""]
    result_lines.append(format_classifications(top_classes))
    result_lines.append("")
    result_lines.append(f"语音检测得分: {speech_score:.1%}")

    return [TextContent(type="text", text="\n".join(result_lines))]


async def transcribe_speech_only(duration: float) -> list[TextContent]:
    """仅进行语音转录"""

    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, record_audio, duration)

    transcription = await loop.run_in_executor(
        None, transcribe_with_whisper, audio
    )

    result_lines = [
        "🎤 语音转录结果 (Whisper)",
        "=" * 30,
        "",
        f"检测语言: {transcription['language']}",
        "",
        "转录内容:",
        f"\"{transcription['text']}\""
    ]

    return [TextContent(type="text", text="\n".join(result_lines))]


async def analyze_audio_file(filepath: str) -> list[TextContent]:
    """分析音频文件"""

    path = Path(filepath)
    if not path.exists():
        return [TextContent(type="text", text=f"❌ 文件不存在: {filepath}")]

    # 加载音频文件
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
    except Exception as e:
        return [TextContent(type="text", text=f"❌ 无法加载音频文件: {e}")]

    duration = len(audio) / 16000

    result_lines = [
        f"📁 音频文件分析: {path.name}",
        f"   时长: {duration:.1f} 秒",
        "=" * 40,
        ""
    ]

    loop = asyncio.get_event_loop()

    # YAMNet 分类
    top_classes, speech_score = await loop.run_in_executor(
        None, classify_with_yamnet, audio
    )

    result_lines.append("📊 声音类型:")
    result_lines.append(format_classifications(top_classes))
    result_lines.append("")

    # Whisper 转录
    if speech_score > 0.2:
        result_lines.append(f"🗣️ 检测到语音 ({speech_score:.1%})")

        transcription = await loop.run_in_executor(
            None, transcribe_with_whisper, audio
        )

        result_lines.append(f"📝 转录 [{transcription['language']}]:")
        result_lines.append(f"   \"{transcription['text']}\"")
    else:
        result_lines.append("ℹ️ 未检测到明显语音内容")

    return [TextContent(type="text", text="\n".join(result_lines))]


async def list_audio_devices() -> list[TextContent]:
    """列出音频设备"""

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default = " (默认)" if i == sd.default.device[0] else ""
            input_devices.append(f"  [{i}] {dev['name']}{default}")

    result = "🎙️ 可用音频输入设备:\n" + "\n".join(input_devices)
    return [TextContent(type="text", text=result)]


# ============== 环境监听功能 ==============

async def start_environment_monitor(
    watch_classes: list[str],
    threshold: float,
    interval: float,
    transcribe: bool
) -> list[TextContent]:
    """启动环境监听"""

    if monitor.is_running:
        return [TextContent(type="text", text="⚠️ 监听器已经在运行中")]

    success = monitor.start(
        watch_classes=watch_classes,
        threshold=threshold,
        interval=interval,
        transcribe=transcribe
    )

    if success:
        watch_info = ", ".join(watch_classes) if watch_classes else "所有声音"
        result_lines = [
            "🎧 环境监听已启动!",
            "=" * 30,
            f"📌 监听目标: {watch_info}",
            f"📊 检测阈值: {threshold:.0%}",
            f"⏱️ 监听间隔: {interval} 秒",
            f"🗣️ 语音转录: {'开启' if transcribe else '关闭'}",
            "",
            "💡 使用 get_monitor_events 查看检测到的事件",
            "💡 使用 stop_monitor 停止监听"
        ]
        return [TextContent(type="text", text="\n".join(result_lines))]
    else:
        return [TextContent(type="text", text="❌ 启动监听失败")]


async def stop_environment_monitor() -> list[TextContent]:
    """停止环境监听"""

    if not monitor.is_running:
        return [TextContent(type="text", text="⚠️ 监听器没有在运行")]

    event_count = len(monitor.events)
    monitor.stop()

    return [TextContent(type="text", text=f"🛑 环境监听已停止\n📊 共记录了 {event_count} 个事件")]


async def get_monitor_events(clear: bool) -> list[TextContent]:
    """获取监听事件"""

    events = monitor.get_events(clear=clear)

    if not events:
        status = "运行中" if monitor.is_running else "已停止"
        return [TextContent(type="text", text=f"📭 暂无事件 (监听器状态: {status})")]

    result_lines = [
        f"📋 检测到的事件 ({len(events)} 个)",
        "=" * 40,
        ""
    ]

    for i, event in enumerate(events, 1):
        result_lines.append(f"🕐 [{event['time']}] 事件 #{i}")

        # 声音类型
        sounds = ", ".join([f"{name} ({score:.0%})" for name, score in event['sounds']])
        result_lines.append(f"   🔊 声音: {sounds}")

        # 语音转录
        if event.get('transcription'):
            trans = event['transcription']
            result_lines.append(f"   🗣️ 语音 [{trans['language']}]: \"{trans['text']}\"")

        result_lines.append("")

    if clear:
        result_lines.append("✅ 事件列表已清空")

    return [TextContent(type="text", text="\n".join(result_lines))]


async def get_monitor_status() -> list[TextContent]:
    """获取监听状态"""

    status = monitor.get_status()

    watch_info = ", ".join(status['watch_classes']) if status['watch_classes'] else "所有声音"
    running_emoji = "🟢" if status['is_running'] else "🔴"

    result_lines = [
        "📊 环境监听器状态",
        "=" * 30,
        f"{running_emoji} 状态: {'运行中' if status['is_running'] else '已停止'}",
        f"📌 监听目标: {watch_info}",
        f"📊 检测阈值: {status['threshold']:.0%}",
        f"⏱️ 监听间隔: {status['interval']} 秒",
        f"📝 已记录事件: {status['event_count']} 个"
    ]

    return [TextContent(type="text", text="\n".join(result_lines))]


# ============== 摄像头和多模态功能 ==============

async def capture_camera_tool(camera_id: int) -> list:
    """拍照工具"""
    result = capture_camera(camera_id)

    if "error" in result:
        return [TextContent(type="text", text=f"❌ 拍照失败: {result['error']}")]

    return [
        ImageContent(type="image", data=result["image_b64"], mimeType="image/jpeg"),
        TextContent(type="text", text=f"📷 已拍摄 ({result['width']}x{result['height']})")
    ]


async def auto_monitor_loop(max_wait_seconds: float, sensitivity: float, camera_id: int) -> list:
    """多模态监控：检测到声音时自动拍照"""

    ignore_classes = {"Silence", "White noise", "Pink noise", "Static"}
    segment_duration = 2
    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        try:
            # 录制音频
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, record_audio, segment_duration)

            # 分析
            top_classes, speech_score = await loop.run_in_executor(
                None, classify_with_yamnet, audio
            )

            # 过滤掉噪音
            meaningful = [
                (name, score) for name, score in top_classes
                if name not in ignore_classes and score >= sensitivity
            ]

            if meaningful:
                top_sound = meaningful[0][0]
                sounds_str = ", ".join(f"{name} ({score:.0%})" for name, score in meaningful[:3])

                # 拍照
                cam_result = capture_camera(camera_id)

                result_text = f"""🎤 检测到声音!
{'=' * 35}
🔊 声音: {sounds_str}
🏆 主要声音: {top_sound}

请根据声音和画面做出回应，然后继续调用 auto_monitor_loop 保持监控！"""

                if "error" not in cam_result:
                    return [
                        ImageContent(type="image", data=cam_result["image_b64"], mimeType="image/jpeg"),
                        TextContent(type="text", text=result_text)
                    ]
                else:
                    return [TextContent(type="text", text=result_text + f"\n\n(拍照失败: {cam_result['error']})")]

            await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"监控循环错误: {e}")
            return [TextContent(type="text", text=f"❌ 错误: {e}")]

    return [TextContent(type="text", text=f"🔇 安静了 {max_wait_seconds} 秒，继续调用 auto_monitor_loop 保持监控")]


async def listen_and_respond(max_wait_seconds: float, sensitivity: float) -> list[TextContent]:
    """等待并监听有意义的声音，返回声音类型和响应建议"""

    ignore_classes = {"Silence", "White noise", "Pink noise", "Static"}
    segment_duration = 2
    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        try:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, record_audio, segment_duration)

            top_classes, speech_score = await loop.run_in_executor(
                None, classify_with_yamnet, audio
            )

            meaningful = [
                (name, score) for name, score in top_classes
                if name not in ignore_classes and score >= sensitivity
            ]

            if meaningful:
                top_sound = meaningful[0][0]
                wait_time = round(time.time() - start_time, 1)
                sounds_str = "\n".join(f"  - {name} ({score:.0%})" for name, score in meaningful)

                result_lines = [
                    "🎧 检测到声音!",
                    "=" * 35,
                    f"⏱️ 等待时间: {wait_time} 秒",
                    "",
                    "🔊 检测到的声音:",
                    sounds_str,
                    "",
                    f"🏆 主要声音: {top_sound}",
                ]

                # 如果有语音，尝试转录
                if speech_score >= 0.15:
                    transcription = await loop.run_in_executor(
                        None, transcribe_with_whisper, audio
                    )
                    if transcription["text"].strip():
                        result_lines.append("")
                        result_lines.append(f"🗣️ 语音内容: \"{transcription['text']}\"")

                return [TextContent(type="text", text="\n".join(result_lines))]

            await asyncio.sleep(0.2)

        except Exception as e:
            return [TextContent(type="text", text=f"❌ 监听错误: {e}")]

    return [TextContent(type="text", text=f"🔇 监听了 {max_wait_seconds} 秒 - 环境很安静")]


# ============== 主入口 ==============

async def main():
    """启动 MCP 服务器"""
    logger.info("启动 Smart Listener MCP 服务器...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
