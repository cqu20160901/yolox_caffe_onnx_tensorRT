# yolox_Caffe_onnx 版本

yolox-nano_caffe_onnx版，后处理用python语言和C++形式进行改写，便于移植不同平台；为了在不同平台之间进行部署（caffe、onnx、tensorRT），对SiLU和Fuse进行了替换。

# 文件结构说明

caffe_yolox：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

onnx_yolox：onnx模型、测试图像、测试结果、测试demo脚本

tensorRT_yolox：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)


# 测试结果

![image](https://github.com/cqu20160901/yolox_caffe_onnx/blob/master/caffe_yolox/result.jpg)

