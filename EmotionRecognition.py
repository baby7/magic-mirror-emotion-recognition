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
from qiniu import BucketManager
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
# 初始化BucketManager
bucket = BucketManager(q)


class EmotionRecognition:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()            # 使用特征提取器get_frontal_face_detector
        self.predictor = dlib.shape_predictor(DATA_PATH)            # dlib的68点模型，使用训练好的特征预测器

    def learning_face(self, path):
        im_read = cv2.imread(path, 1)
        face_list = self.detector(cv2.cvtColor(im_read, cv2.COLOR_RGB2GRAY), 0)   # 对灰度图像使用检测器检测
        if len(face_list) != 0:                                     # 检测器检测到人脸
            for i in range(len(face_list)):                         # 获取68个特征点
                for k, d in enumerate(face_list):                   # 获取索引和数据，k为索引，d为对象
                    box_width = d.right() - d.left()                # 计算人脸框边长
                    appear = self.predictor(im_read, d)             # 获取68个特征点坐标
                    mouth_higth = (appear.part(66).y - appear.part(62).y) / box_width
                    # 眉毛直线拟合列表
                    brow_x = []
                    brow_y = []
                    for j in range(17, 21):
                        brow_x.append(appear.part(j).x)
                        brow_y.append(appear.part(j).y)
                    z1 = np.polyfit(np.array(brow_x), np.array(brow_y), 1)
                    brow_k = -round(z1[0], 3)                       # 拟合成一条直线
                    eye_hight = ((appear.part(41).y - appear.part(37).y + appear.part(40).y - appear.part(38).y +
                               appear.part(47).y - appear.part(43).y + appear.part(46).y - appear.part(44).y)
                                 / 4) / box_width                   # 获取眼睛睁开大小
                    upload_qiniu(path)                              # 上传至图床
                    if round(mouth_higth >= 0.03):
                        if eye_hight >= 0.056:
                            send_face_data_save(path, "3")          # 惊讶
                            return json.dumps({"state": "success", "result": "amazing"})
                        else:
                            send_face_data_save(path, "4")          # 开心
                            return json.dumps({"state": "success", "result": "happy"})
                    else:
                        if brow_k <= -0.3:
                            send_face_data_save(path, "1")          # 生气
                            return json.dumps({"state": "success", "result": "angry"})
                        else:
                            send_face_data_save(path, "2")          # 正常
                            return json.dumps({"state": "success", "result": "nature"})
                break                                               # 只识别一张人脸
        return json.dumps({"state": "error", "result": "no face"})  # 未识别到人脸


app = Flask(__name__)

my_face = EmotionRecognition()
file_paths = ""


# 上传到七牛云
def upload_qiniu(path):
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, path.replace("D:/install/face_data/", ""), 3600)
    # 要上传文件的本地路径
    ret, info = put_file(token, path.replace("D:/install/face_data/", ""), path)
    print(info)


# 删除处理完成的文件
def delete_qiniu(path):
    ret, info = bucket.delete(bucket_name, path.replace("D:/install/face_data/", ""))
    print(info)


# 进行后端人脸情绪保存
def send_face_data_save(path, state):
    url = 'http://62.234.97.198:8005/admin/photo/faceRegistration'
    data = {
        'url': "https://magic-mirror-media.baby7blog.com/" + path.replace("D:/install/face_data/", ""),
        'state': state
    }
    print(data)
    r = requests.post(url, json=data, headers={'Content-Type': 'application/json;charset=UTF-8'})
    result = r.json()
    print(result)
    delete_qiniu(path)


# 定义路由
@app.route("/emotionRecognition", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录
    file_path = r'D:/install/face_data/'
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
