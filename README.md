# MCP Smart Listener 🎧📷

让 Claude 能够"听到"声音和"看到"画面的 MCP 服务器！

结合多个强大的模型:
- **YAMNet**: Google 的音频事件分类模型，识别 521 种声音类型
- **Whisper**: OpenAI 的语音识别模型，将语音转录为文字
- **OpenCV**: 摄像头捕捉，让 Claude 看见你

## 功能

### 基础听音功能

| 工具 | 描述 |
|------|------|
| `listen` | 智能听音：自动识别声音类型，如果有语音则转录 |
| `classify_sound` | 仅识别声音类型（狗叫、音乐、敲门声等） |
| `transcribe_speech` | 仅转录语音内容 |
| `analyze_file` | 分析音频文件 |
| `list_audio_devices` | 列出可用麦克风 |

### 环境监听功能

| 工具 | 描述 |
|------|------|
| `start_monitor` | 启动环境常驻监听，检测到指定声音时记录事件 |
| `stop_monitor` | 停止环境监听 |
| `get_monitor_events` | 获取监听到的事件列表 |
| `get_monitor_status` | 获取监听器状态 |

### 多模态功能 (声音+视觉)

| 工具 | 描述 |
|------|------|
| `capture_camera` | 拍一张照片让 Claude 分析（表情、动作、物体、场景）📷 |
| `auto_monitor_loop` | 【多模态监控】听到声音时自动拍照！🎤📷 |
| `listen_and_respond` | 等待并监听有意义的声音，忽略静音和噪音 |

## 安装

### 1. 创建虚拟环境

```bash
cd mcp-smart-listener
python -m venv venv

# Windows (CMD)
venv\Scripts\activate

# Windows (PowerShell) - 如果报错，用 CMD
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 FFmpeg (Whisper 需要)

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## 配置 Claude Desktop

编辑 `%APPDATA%\Claude\claude_desktop_config.json` (Windows) 或 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "smart-listener": {
      "command": "D:\\Claude code\\mcp-smart-listener\\venv\\Scripts\\python.exe",
      "args": ["D:\\Claude code\\mcp-smart-listener\\smart_listener.py"]
    }
  }
}
```

## 使用示例

### 智能听音

```
你: 帮我听一下周围有什么声音
Claude: [使用 listen 工具]

🎧 智能听音分析结果
==============================

📊 检测到的声音类型:
  Speech: 72.3% ████████████████
  Music: 45.1% █████████
  Keyboard typing: 23.4% ████

🗣️ 语音检测得分: 72.3%

✅ 检测到语音，正在转录...

📝 转录结果 [zh]:
   "今天天气真不错啊"
```

### 多模态监控

```
你: 用 auto_monitor_loop 监控我，检测到声音就拍照回应我
Claude: [使用 auto_monitor_loop 工具]

🎤 检测到声音!
===================================
🔊 声音: Coughing (45%), Speech (30%)
🏆 主要声音: Coughing

[拍摄的照片]

Claude: 看到你咳嗽了，要不要休息一下？需要我帮你查一下附近的药店吗？
```

### 环境监听

```
你: 帮我监听有没有狗叫声
Claude: [使用 start_monitor 工具，watch_classes=["dog"]]

🎧 环境监听已启动!
==============================
📌 监听目标: dog
📊 检测阈值: 30%
⏱️ 监听间隔: 2 秒
🗣️ 语音转录: 开启
```

## 模型选择

### Whisper 模型大小

在 `smart_listener.py` 中可以修改:

```python
whisper_model = whisper.load_model("base")  # 修改这里
```

| 模型 | 大小 | 速度 | 准确率 |
|------|------|------|--------|
| tiny | 39M | 最快 | 一般 |
| base | 74M | 快 | 较好 |
| small | 244M | 中等 | 好 |
| medium | 769M | 慢 | 很好 |
| large | 1550M | 最慢 | 最佳 |

## 声音类别示例

YAMNet 可以识别的声音类型（共 521 种）:

- **人声**: Speech, Singing, Laughter, Crying, Cough, Sneeze, Yawn, Sigh
- **动物**: Dog bark, Cat meow, Bird song, Rooster crow
- **音乐**: Music, Guitar, Piano, Drum, Violin
- **环境**: Rain, Thunder, Wind, Water, Fire
- **机械**: Car, Engine, Horn, Alarm, Siren
- **家居**: Door knock, Doorbell, Keyboard, Phone ring
- ... 更多

## 配合 TTS 使用

可以配合 [mcp-tts](../mcp-tts) 实现完整的语音交互:

```
用户: 用 auto_monitor_loop 监听，检测到声音后用 TTS 回应我

[Claude 检测到咳嗽声 + 拍到照片]

Claude: [调用 speak_minimax] "听到你咳嗽了，看起来有点累，要不要休息一下？"
```

## 故障排除

### 录音权限问题

确保应用有麦克风访问权限。

### 摄像头权限问题

确保应用有摄像头访问权限。如果 `capture_camera` 失败，检查摄像头是否被其他程序占用。

### PowerShell 执行策略

如果在 PowerShell 中无法激活虚拟环境，使用 CMD 或运行:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CUDA/GPU 加速

如果有 NVIDIA GPU，安装 CUDA 版本的 TensorFlow 和 PyTorch 可以加速推理:

```bash
pip install tensorflow[and-cuda]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 模型下载慢

首次运行会下载模型，可能需要一些时间:
- YAMNet: ~20MB
- Whisper base: ~140MB

## License

MIT

---

<p align="center">
  <a href="https://buymeacoffee.com/neige_neige">
    <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-yellow?style=flat-square&logo=buy-me-a-coffee">
  </a>
</p>

<p align="center">
  <sub>Built with 🌀 <a href="https://github.com/anthropics/claude-code">Claude Code</a></sub>
</p>
