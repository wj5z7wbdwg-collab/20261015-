# -*- coding: utf-8 -*-
"""
猜拳游戏 Web 服务

- 提供单页网页（静态目录 web/）
- 通过 WebSocket 接收浏览器发送的摄像头帧，返回 ONNX 模型的实时手势识别结果
- 监听 0.0.0.0，允许局域网内其他设备访问

运行：
    python web_server.py
然后本机打开  http://127.0.0.1:8000
局域网设备打开  http://<本机局域网IP>:8000   （启动时会打印）
"""

import asyncio
import base64
import json
import os
import socket
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
ONNX_PATH = os.path.join(BASE_DIR, "garbage-classification.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# 与实时识别程序一致的阈值与平滑窗口
CONF_THRESHOLD = 0.60
SMOOTH_WINDOW = 5

# ---------------------------------------------------------------- 模型与标签


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip().split(" ")[1] for line in f.readlines()]


def load_session():
    # 读取字节再创建会话，兼容含中文路径
    with open(ONNX_PATH, "rb") as f:
        return ort.InferenceSession(f.read(), providers=["CPUExecutionProvider"])


LABELS = load_labels()
SESSION = load_session()
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess_bgr(frame_bgr):
    """与训练一致：224x224、BGR->RGB、归一化到 [-1,1]、加 batch 维。"""
    img = cv2.resize(frame_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img = (img / 127.5) - 1
    return np.expand_dims(img, axis=0)


def infer(frame_bgr):
    x = preprocess_bgr(frame_bgr)
    pred = SESSION.run(None, {INPUT_NAME: x})[0][0]
    idx = int(np.argmax(pred))
    return LABELS[idx], float(pred[idx]), pred.tolist()


# ---------------------------------------------------------------- FastAPI 应用

app = FastAPI(title="RPS Web Game")


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# 手势实拍图（前端用于“获胜手势”展示）
DATASET_DIRS = {"石头": "石头", "剪刀": "剪刀", "布": "布"}
for g, d in DATASET_DIRS.items():
    p = os.path.join(BASE_DIR, d)
    if os.path.isdir(p):
        app.mount(f"/dataset/{g}", StaticFiles(directory=p), name=f"dataset_{g}")

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ---------------------------------------------------------------- WebSocket

class ConnectionState:
    def __init__(self):
        self.history = deque(maxlen=SMOOTH_WINDOW)
        self.last_time = 0.0


@app.websocket("/ws")
async def ws_infer(websocket: WebSocket):
    await websocket.accept()
    state = ConnectionState()
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue

            b64 = payload.get("frame")
            if not b64:
                continue

            # 去掉 data:image/jpeg;base64, 前缀
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                data = base64.b64decode(b64)
            except Exception:
                continue
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            label, conf, probs = infer(frame)
            state.history.append(label)
            final_label = max(set(state.history), key=state.history.count)

            await websocket.send_text(json.dumps({
                "label": final_label,
                "raw_label": label,
                "confidence": round(conf, 4),
                "probs": {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))},
                "threshold": CONF_THRESHOLD,
                "ts": int(time.time() * 1000),
            }))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------- 启动

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    import threading
    import uvicorn

    port = 8888
    ip = lan_ip()
    cert = os.path.join(BASE_DIR, "cert.pem")
    key = os.path.join(BASE_DIR, "key.pem")
    use_https = os.path.exists(cert) and os.path.exists(key)
    scheme = "https" if use_https else "http"

    # ---- HTTP 自动跳转：http://<ip>:8000 -> https://<ip>:8000 ----
    # 另起一个轻量 HTTP 服务(8001)，只做 301 重定向，避免用户忘记输 https://
    if use_https:
        from fastapi import Request
        from fastapi.responses import RedirectResponse

        redirect_app = FastAPI(title="HTTP->HTTPS Redirect")

        @redirect_app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
        async def _redirect(full_path: str, request: Request):
            host = request.headers.get("host", f"{ip}:8001").split(":")[0]
            target = f"https://{host}:{port}/{full_path}"
            if request.url.query:
                target += "?" + request.url.query
            return RedirectResponse(url=target, status_code=301)

        def run_redirect():
            uvicorn.run(redirect_app, host="0.0.0.0", port=8001, log_level="warning")

        threading.Thread(target=run_redirect, daemon=True).start()

    print("=" * 60)
    print("猜拳游戏 Web 服务已启动")
    print(f"  本机访问:    {scheme}://127.0.0.1:{port}")
    print(f"  局域网访问:  {scheme}://{ip}:{port}   <-- 分享给同一 WiFi 下的其他人")
    if use_https:
        print(f"  HTTP 跳转:   http://{ip}:8001  会自动跳到 https://{ip}:{port}")
        print("  本机已信任证书；其他设备首次打开点「高级」->「继续前往」即可。")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning",
                ssl_certfile=cert if use_https else None,
                ssl_keyfile=key if use_https else None)
