# yolox onnx 版本

对yolox导出onnx，导出onnx时需要将后处理部分关闭，模型forward完成后将卷积层的输出结果进行了faltten，然后对后处理用python语言+C语言的风格进行进行了重写，便于不同平台进行移植。
