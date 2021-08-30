import requests
import json
import base64
import os
import numpy as np
import math
import cv2
import sys
import time

from PIL import ImageFont, Image, ImageDraw

API_CATEGORY = os.environ.get("API_CATEGORY", "cv")
API_NAME = os.environ.get("API_NAME", "lp_detect")
API_VERSION = os.environ.get("API_VERSION", "1.0")
DEUBG_MODE = os.environ.get("DEBUG_MODE", "True")

API_URI = '/%s/%s/%s' % (API_CATEGORY, API_NAME, API_VERSION)
Healthy_URI = '%s/%s' % (API_URI, "healthy")

# print(API_URI)

def pic2base64(image_path):

    with open(image_path, 'rb') as f:
        image = f.read()
    image_base64 = base64.b64encode(image)
    image_base64 = image_base64.decode()

    return image_base64


def post_request(imageId,image_path,url=None):

    url_ = 'http://127.0.0.1:8080'+API_URI
    base64Data = pic2base64(image_path)
    format = os.path.splitext(image_path)[-1].replace(".","")
    url = ""

    data = {"imageId": imageId,
            "base64Data": base64Data,
            "format": format,
            "url": url}

    data = json.dumps(data)
    res = requests.post(url_, data=data).text
    res = json.loads(res)

    resut_lp_chines = res["result_lp_chines"]
    result_lp_xys = res["result_lp_xys"]
    img = cv2.imread(image_path)

    if result_lp_xys:


        font = ImageFont.truetype('SimHei.ttf', 20, encoding="utf-8")
        for (x1,y1,x2,y2),sequence in zip(result_lp_xys,resut_lp_chines):
            cv2.rectangle(img,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
            pilImg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pilImg)
            print(sequence)
            draw.text((x1,y1-20), sequence, (0, 200, 100), font=font)
            img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
            # cv2.putText(img, sequence, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.imwrite("result.jpg",img)
    cv2.waitKey(0)


if __name__ == '__main__':
    post_request("00001","image_test/image/test4.jpg")
