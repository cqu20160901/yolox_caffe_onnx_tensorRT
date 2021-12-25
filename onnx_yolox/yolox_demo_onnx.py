import cv2
import numpy as np
import onnxruntime as ort
from math import exp

CLASSES = ('o1:', 'o2:')

meshgrid = []

class_num = 2
headNum = 6
strides = [8, 16, 32, 64, 128, 256]
mapSize = [[52, 92], [26, 46], [13, 23], [7, 12], [4, 6], [2, 3]]
nmsThresh = 0.45
objectThresh = [0.35, 0.35]

input_imgH = 414
input_imgW = 736


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j)
                meshgrid.append(i)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    gridIndex = -2

    for index in range(headNum):
        reg = output[index * 3 + 0]
        ce = output[index * 3 + 1]
        cls = output[index * 3 + 2]

        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):

                gridIndex += 2

                ce_val = sigmoid(ce[h * mapSize[index][1] + w])

                for cl in range(class_num):
                    cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * ce_val

                    if cls_val > objectThresh[cl]:
                        cx = (meshgrid[gridIndex + 0] + reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        cy = (meshgrid[gridIndex + 1] + reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        xf = exp(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        yf = exp(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]

                        xmin = (cx - xf / 2) * scale_w
                        ymin = (cy - yf / 2) * scale_h
                        xmax = (cx + xf / 2) * scale_w
                        ymax = (cy + yf / 2) * scale_h

                        if xmin >= 0 and ymin >= 0 and xmax <= img_w and ymax <= img_h:
                            box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax)
                            detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def preprocess(src):
    im = cv2.resize(src, (input_imgW, input_imgH))
    img = im.astype(np.float32)

    return img


def detect(imgfile, model_path):

    orig = cv2.imread(imgfile)
    img_h, img_w = orig.shape[:2]

    img = preprocess(orig)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    ort_session = ort.InferenceSession(model_path)
    res = (ort_session.run(None, {'data': img}))

    out = []
    for i in range(len(res)):
        out.append(res[i])

    predbox = postprocess(out, img_h, img_w)

    print('obj num is :', len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./result_onnx.jpg', orig)
    # cv2.imshow("test", orig)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main .... ')
    GenerateMeshgrid()
    detect('./test.jpg', './yolox_736x414.onnx')
