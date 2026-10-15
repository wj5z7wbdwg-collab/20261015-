# ✊ 手势猜拳 · AI Rock Paper Scissors

基于深度学习手势识别的**局域网在线猜拳游戏**。摄像头实时识别玩家手势（石头 / 剪刀 / 布 / 背景），计算机随机出拳对战，实时计分。

## ✨ 功能

- 📷 浏览器获取摄像头视频流，WebSocket 推送帧到后端实时推理（ONNX）
- 🎯 4 分类识别：石头 / 剪刀 / 布 / 背景，5 帧平滑防抖 + 置信度阈值
- 🤖 人机对战：3-2-1 倒计时、随机出拳 + 20% 针对性策略、胜负判定、计分与连胜统计
- 🖼️ 识别后自动提示"能赢你的手势"（展示数据集实拍图）
- 🔊 WebAudio 合成音效、霓虹科技风 UI、对战历史记录
- 🌐 局域网 HTTPS 服务，同一 WiFi 下其他设备直接访问即可对战

## 🚀 快速开始

```bash
pip install -r requirements.txt
python web_server.py
```

启动后访问控制台打印的地址：

- 本机：`https://127.0.0.1:8888`
- 局域网：`https://<本机IP>:8888`

> 首次打开若提示证书不受信任：点「高级 → 继续前往」即可；
> Windows 本机可右键 `cert.pem` → 安装证书 → 「受信任的根证书颁发机构」，消除警告。
> 摄像头功能要求安全上下文（HTTPS），HTTP 访问无法调用 getUserMedia。

## 📁 项目结构

```
├── web_server.py        # FastAPI 后端：静态页 + /ws 实时推理 + HTTP→HTTPS 跳转
├── realtime_game.py     # 桌面版：OpenCV 摄像头实时识别（作业版本）
├── predict_image.py     # 单张图片识别示例
├── convert_model.py     # keras_model.h5 -> onnx 转换脚本
├── keras_model.h5       # Teachable Machine 训练的手势识别模型（Keras2）
├── garbage-classification.onnx  # 转换后的推理模型（224x224, 4 分类）
├── labels.txt           # 标签：0 剪刀 / 1 石头 / 2 布 / 3 背景
├── 石头/ 剪刀/ 布/ 背景/  # 自采数据集（每类 200+ 张）
└── web/                 # 前端（index.html + style.css + app.js）
```

## 🧠 模型说明

- 训练：Teachable Machine 图像分类，导出 `keras_model.h5`
- 转换：`convert_model.py`（需 `tensorflow` + `tf-keras` + `tf2onnx`；
  注意 TensorFlow 在含中文路径下会报错，需复制到英文目录转换）
- 推理：`onnxruntime` CPU，输入 224×224，归一化到 [-1, 1]

## ⚙️ 技术栈

FastAPI · Uvicorn · WebSocket · ONNX Runtime · OpenCV · 原生 HTML/CSS/JS · WebRTC · WebAudio
