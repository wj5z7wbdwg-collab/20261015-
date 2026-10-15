import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# --- 1. 初始化：读取标签与加载模型 ---
with open('labels.txt', 'r', encoding='utf-8') as f:
    GARBAGE_LABELS = [line.strip().split(' ')[1] for line in f.readlines()]

onnx_model_path = "garbage-classification.onnx"
ort_session = ort.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name

try:
    font = ImageFont.truetype("simhei.ttf", 20)
except IOError:
    font = ImageFont.load_default()
    print("警告：未找到中文字体'simhei.ttf'，将使用默认字体，中文可能无法显示。")

# --- 2. 读取单张图片 ---
# 将下面的路径修改为待识别图片的路径
image_path = "test.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print(f"错误：无法读取图片 {image_path}，请检查路径是否正确。"); exit()

# --- 3. 图像预处理 ---
# 将图像尺寸调整为模型需要的224x224
img_resized = cv2.resize(frame, (224, 224))
# OpenCV读入的是BGR格式，需转为RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# 转为float32并归一化到[-1, 1]
img_normalized = (img_rgb.astype('float32') / 127.5) - 1
# 增加批次维度，模型期望的输入是 (N, H, W, C)
input_tensor = np.expand_dims(img_normalized, axis=0)

# --- 4. 执行推理 ---
prediction = ort_session.run(None, {input_name: input_tensor})[0]

# --- 5. 解析结果 ---
predicted_index = np.argmax(prediction)
predicted_label = GARBAGE_LABELS[predicted_index]
confidence = np.max(prediction)

print("各类别置信度：")
for name, score in zip(GARBAGE_LABELS, prediction[0]):
    print(f"  {name}: {score:.4f}")
print(f"预测结果: {predicted_label} ({confidence:.2f})")

# --- 6. 在图片上绘制中文结果并显示 ---
result_text = f"预测结果: {predicted_label} ({confidence:.2f})"
display_frame = frame.copy()
pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_img)
draw.text((10, 30), result_text, font=font, fill=(0, 255, 0))
display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 保存标注后的图片，并在窗口中显示
cv2.imwrite("result.jpg", display_frame)
cv2.imshow("Single Image Garbage Classification", display_frame)
print("按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()