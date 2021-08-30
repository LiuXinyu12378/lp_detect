import cv2
from flask import Flask, jsonify, request
import numpy as np
from app_ import detect_license_plate, STATUS_INFO

app = Flask(__name__)



@app.route('/leez/info',methods=["GET"])
def info():

    if STATUS_INFO :
        data = {"msg": "LeezStudio benchmark protocol info.", "result": {}, "status": 0}
        return jsonify(data),200

    else:
        data = {"msg": "LeezStudio benchmark protocol not readly.", "result": {}, "status": 500}
        return jsonify(data),500



@app.route("/leez/benchmark/<channelid>",methods=["GET"])
def benchmark(channelid):
    BENCHMARK = True
    if BENCHMARK:
        data = {
            "msg": "LeezStudio benchmark protocol.",
            "result": {
                "LoadAverage": "0.00, 0.04, 0.14",
                "CPU": 0,
                "FreeMaxMem": 1234,
                "FreeMem": 234,
                "GPU_Rate": 0,
                "GPU_Mem": 0,
                "GPU_Mhz": 0
            },
            "status": 0
        }
        return jsonify(data),200
    else:
        data = {"msg":"LeezStudio benchmark protocol not readly.","result":{},"status":500}
        return jsonify(data),500



@app.route("/leez/img/<channelid>",methods=["POST"])
def img(channelid):

    try:
        f = request.files['img']

        fileformat = f.filename.split(".")[-1]

        if fileformat not in ["jpeg","jpg","png"]:
            data = {"msg": "Decode image error.", "result": {}, "status": 500}
            return jsonify(data), 500

        result_lp_chines, result_lp_xys = detect_license_plate(f.read())

        print(result_lp_chines,result_lp_xys)

        data = {
                "result":[
                    {
                        "task":0,
                        "label":result_lp_chines,
                        "objId":"1",
                        "score":0.9,
                        "ts":0,
                        "bbox":result_lp_xys
                    }
                ],
                "status":0,
                "filename":f.filename
            }
        return jsonify(data),200
    except EnvironmentError as e:
        data = {"msg":"Process image error.","result":{},"status":404}
        return jsonify(data),404



@app.route("/leez/teardown",methods=["GET"])
def teradown():

    data = {"msg":"LeezStudio benchmark protocol info.","result":{},"status":0}
    return jsonify(data),200




if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8085)