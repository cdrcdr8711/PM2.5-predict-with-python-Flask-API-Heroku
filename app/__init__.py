# -*- coding: UTF-8 -*-
import numpy as np
import app.model as model

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/test', methods=['GET'])
def getResult():
    input = np.array([[5.5, 2.4, 2.7, 1.]])
    result = model.predict(input)
    return jsonify({'result': str(result)})


@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1 = insertValues['season']
    x2 = insertValues['DEWP']
    x3 = insertValues['HUMI']
    x4 = insertValues['PRES']
    x5 = insertValues['TEMP']
    x6 = insertValues['Iws']
    x7 = insertValues['precipitation']
    x8 = insertValues['Iprec']
    x9 = insertValues['weekday']
    input = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9]])

    result = model.predict(input)

    return jsonify({'return': str(result)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
