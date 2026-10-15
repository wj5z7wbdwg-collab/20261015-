# -*- coding: utf-8 -*-
"""
猜拳游戏 —— 摄像头实时识别程序

功能：
    1. 调用摄像头，实时识别画面中的 石头 / 剪刀 / 布 / 背景 手势；
    2. 左上角显示识别结果与置信度；
    3. 右上角显示能赢过该手势的“获胜手势”名称与图片。

运行前需先由 keras_model.h5 生成 garbage-classification.onnx（见 convert_model.py）：
    pip install tf-keras
    $env:TF_USE_LEGACY_KERAS="1"   # 该 h5 为 Keras 2 格式，需用 legacy Keras 加载
    python convert_model.py
注意：TensorFlow 在含中文的路径下会报错，建议先把 keras_model.h5 复制到纯英文目录转换，
再把生成的 garbage-classification.onnx 拷回本目录。
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 能赢过当前手势的手势：石头->布、剪刀->石头、布->剪刀
WIN_AGAINST = {"石头": "布", "剪刀": "石头", "布": "剪刀"}

# 右上角展示的获胜手势图片，直接取自采集的数据集实拍图
WIN_IMAGE_PATH = {
    "石头": os.path.join(BASE_DIR, "石头", "WIN_20261015_08_54_52_Pro.jpg"),
    "剪刀": os.path.join(BASE_DIR, "剪刀", "WIN_20261015_08_55_44_Pro.jpg"),
    "布": os.path.join(BASE_DIR, "布", "WIN_20261015_08_50_48_Pro.jpg"),
}

# 中文字体（Windows 常用字体，按顺序尝试）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

CONF_THRESHOLD = 0.60   # 置信度低于该值视为“识别中”
SMOOTH_WINDOW = 5       # 连续多帧投票，避免画面抖动
WIN_IMG_SIZE = (222, 125)


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("警告：未找到中文字体，中文可能无法正常显示。")
    return ImageFont.load_default()


def load_labels():
    with open(os.path.join(BASE_DIR, "labels.txt"), "r", encoding="utf-8") as f:
        return [line.strip().split(" ")[1] for line in f.readlines()]


def imread_unicode(path):
    """读取图片，兼容含中文的路径（cv2.imread 在中文路径下会失败）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_win_images():
    """预加载右上角要展示的三张获胜手势图片，统一缩放到固定尺寸。"""
    images = {}
    for gesture, path in WIN_IMAGE_PATH.items():
        img = imread_unicode(path)
        if img is None:
            print(f"警告：无法读取手势图片 {path}")
            continue
        images[gesture] = cv2.resize(img, WIN_IMG_SIZE)
    return images


def preprocess(frame):
    """与训练时一致：缩放到 224x224、BGR->RGB、归一化到 [-1, 1]、增加批次维度。"""
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img = (img / 127.5) - 1
    return np.expand_dims(img, axis=0)


def put_text(frame, text, position, font, color):
    """用 PIL 在 OpenCV 画面上绘制中文文字。"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_overlay(frame, gesture, confidence, win_images, font_big, font_small):
    """左上角显示识别结果，右上角显示能赢过该手势的获胜手势与图片。"""
    # 左上角：识别结果
    if confidence < CONF_THRESHOLD:
        result_text = "识别结果：识别中..."
    else:
        result_text = f"识别结果：{gesture}  {confidence:.2f}"
    frame = put_text(frame, result_text, (20, 20), font_big, (0, 255, 0))

    # 右上角：获胜手势与图片（背景类不显示）
    if gesture in WIN_AGAINST and confidence >= CONF_THRESHOLD:
        win_gesture = WIN_AGAINST[gesture]
        img_w, img_h = WIN_IMG_SIZE
        x, y = frame.shape[1] - img_w - 20, 20
        if win_gesture in win_images:
            frame[y:y + img_h, x:x + img_w] = win_images[win_gesture]
        frame = put_text(frame, f"获胜手势：{win_gesture}",
                         (x, y + img_h + 8), font_small, (0, 215, 255))
    return frame


def open_camera(index=0):
    """依次尝试不同后端打开摄像头，返回已打开且能读到画面的 VideoCapture。"""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def main():
    onnx_path = os.path.join(BASE_DIR, "garbage-classification.onnx")
    if not os.path.exists(onnx_path):
        print("错误：未找到 garbage-classification.onnx，请先运行 convert_model.py 转换模型。")
        return

    labels = load_labels()
    win_images = load_win_images()
    font_big = load_font(34)
    font_small = load_font(24)

    # 读取模型字节再创建会话，兼容含中文的路径
    session = ort.InferenceSession(open(onnx_path, "rb").read(),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = open_camera()
    if cap is None:
        print("错误：无法打开摄像头，请检查摄像头是否被其它程序占用、系统隐私设置是否允许应用访问摄像头。")
        return

    history = deque(maxlen=SMOOTH_WINDOW)
    started = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("错误：读取摄像头画面失败。")
            break
        if not started:
            print("摄像头已启动，按 ESC 或 q 退出。")
            started = True

        prediction = session.run(None, {input_name: preprocess(frame)})[0][0]
        history.append(int(np.argmax(prediction)))

        # 多帧投票，取出现次数最多的类别作为稳定结果
        final_index = max(set(history), key=history.count)
        gesture = labels[final_index]
        confidence = float(prediction[final_index])

        frame = draw_overlay(frame, gesture, confidence, win_images, font_big, font_small)
        cv2.imshow("RPS Real-time Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
