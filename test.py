import requests

# # API地址
# url = "http://3cw0735693.qicp.vip/faceRecognition"
# # 图片地址
# file_path = "test.jpg"
# # 图片名
# file_name = file_path.split('/')[-1]
# # 二进制打开图片
# file = open(file_path, 'rb')
# # 拼接参数
# files = {'file': (file_name, file, 'image/jpg')}
# # 发送post请求到服务器端
# r = requests.post(url, files=files)
# # 获取服务器返回的信息
# result = r.content
# # 字节转换成图片
# print(result)


url = 'http://62.234.97.198:8005/admin/corpus/chat'
data = {
    'userId': 1,
    'question': '房地产'
}
r = requests.post(url, data=data)
result = r.json()
if result['code'] is 0:
    print(result['data'][0])
