#!Anaconda/anaconda/python
# coding: utf-8

"""
从视屏中识别人脸，并实时标出面部特征点
"""

from flask import request
from flask import Flask
import os
import dlib
import numpy as np
import cv2
from qiniu import Auth, put_file
import requests
import json


DATA_PATH = "shape_predictor_68_face_landmarks.dat"
IMAGE_PATH = "test.jpg"
access_key = 'QHqjfdZsbHZaAvogeFGM8cTrurShs7UmN5LIu-iZ'
secret_key = 'yDdNmZWUEzs2zsk46O6O5-WbZ0x1lF2r3neJ7xbO'
# 构建鉴权对象
q = Auth(access_key, secret_key)
# 要上传的空间
bucket_name = 'magic-mirror-media'


class EmotionRecognition:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # 使用特征提取器get_frontal_face_detector
        self.predictor = dlib.shape_predictor(DATA_PATH)  # dlib的68点模型，使用训练好的特征预测器

    def learning_face(self, path):
        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []
        im_rd = cv2.imread(path, 1)
        img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  # 取灰度
        faces = self.detector(img_gray, 0)  # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
        # 如果检测到人脸
        if len(faces) != 0:
            for i in range(len(faces)):  # 对每个人脸都标出68个特征点
                for k, d in enumerate(faces):  # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    face_width = d.right() - d.left()  # 计算人脸热别框边长
                    shape = self.predictor(im_rd, d)  # 使用预测器得到68点数据的坐标
                    # 分析任意n点的位置关系来作为表情识别的依据
                    mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width  # 嘴巴张开程度
                    # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
                    brow_sum = 0  # 高度之和
                    frown_sum = 0  # 两边眉毛距离之和
                    for j in range(17, 21):
                        brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                        frown_sum += shape.part(j + 5).x - shape.part(j).x
                        line_brow_x.append(shape.part(j).x)
                        line_brow_y.append(shape.part(j).y)
                    tempx = np.array(line_brow_x)
                    tempy = np.array(line_brow_y)
                    z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
                    brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
                    # 眼睛睁开程度
                    eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                               shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
                    eye_hight = (eye_sum / 4) / face_width
                    upload_qiniu(path)
                    if round(mouth_higth >= 0.03):
                        if eye_hight >= 0.056:
                            send_face_data_save(path, "3")
                            return json.dumps({"state": "success", "result": "amazing"})   # 惊讶
                        else:
                            send_face_data_save(path, "4")
                            return json.dumps({"state": "success", "result": "happy"})     # 开心
                    else:
                        if brow_k <= -0.3:
                            send_face_data_save(path, "1")
                            return json.dumps({"state": "success", "result": "angry"})     # 生气
                        else:
                            send_face_data_save(path, "2")
                            return json.dumps({"state": "success", "result": "nature"})    # 正常
                break                                                                      # 只识别一张人脸
        return json.dumps({"state": "error", "result": "no face"})                         # 未识别到人脸


app = Flask(__name__)

my_face = EmotionRecognition()
file_paths = ""


# 上传到七牛云
def upload_qiniu(path):
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, path, 3600)
    # 要上传文件的本地路径
    ret, info = put_file(token, path.replace("C:/Users/xiang/Downloads/", ""), path)
    print(info)


# 进行后端人脸情绪保存
def send_face_data_save(path, state):
    path = path.replace("C:/Users/xiang/Downloads/", "")
    url = 'http://62.234.97.198:8005/admin/photo/faceRegistration'
    data = {
        'url': "https://magic-mirror-media.baby7blog.com/" + path,
        'state': state
    }
    print(data)
    r = requests.post(url, json=data, headers={'Content-Type': 'application/json;charset=UTF-8'})
    result = r.json()
    print(result)


# 定义路由
@app.route("/emotionRecognition", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录
    file_path = r'C:/Users/xiang/Downloads/'
    if upload_file:
        # 地址拼接
        file_path_add = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_path_add)
        # 进行人脸识别
        result = my_face.learning_face(file_path_add)
        # 返回结果
        return result
    else:
        return json.dumps({"state": "error", "result": "no photo"})


if __name__ == "__main__":
    app.run()
