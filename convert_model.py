import tensorflow as tf
import tf2onnx
import json

# 定义输入和输出文件名
keras_model_path = "keras_model.h5"
output_onnx_path = "garbage-classification.onnx"

try:
    # 加载Keras模型
    model = tf.keras.models.load_model(keras_model_path)

# 将Keras模型转换为ONNX格式
# spec：指定模型的输入签名，对于本项目导出的图像模型，通常是(None, 224, 224, 3)
# opset：ONNX的版本号，13是较为通用的版本
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input")], opset=13)

    # 保存ONNX模型
    with open(output_onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"模型转换成功！已将 '{keras_model_path}' 转换为 '{output_onnx_path}'")

except Exception as e:
    print(f"模型转换失败：{e}")
    print("请确保 'keras_model.h5' 文件与此脚本在同一个文件夹下。")