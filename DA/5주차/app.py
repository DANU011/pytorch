# import time
# import csv
# from flask import Flask, request
# import numpy as np
# import joblib
#
# app = Flask(__name__)
#
# # KNN 모델 경로 지정
# knn_model_file = "./model/model_knn.pkl"
# scaler_file = "./model/scaler_knn.pkl"
#
# # KNN 모델 로드
# knn_model = joblib.load(knn_model_file)
#
# # Scaler 로드
# scaler = joblib.load(scaler_file)
#
# # 데이터 전처리 함수
# def preprocess_data(data):
#     scaled_data = scaler.transform(np.array(data).reshape(1, -1))
#     return scaled_data
#
#
# # CSV 파일 레코드 한 줄씩 처리 함수
# def process_record(record):
#     # 선택한 필드 추출
#     selected_data = [float(record[0]), float(record[1]), float(record[2]), float(record[3]), float(record[4])]
#
#     # 데이터 전처리
#     preprocessed_data = preprocess_data(selected_data)
#
#     # 예측
#     prediction = knn_model.predict(preprocessed_data)
#
#     # 예측 결과 반환
#     return prediction
#
# # /predict 엔드포인트
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return "No file provided"
#
#     file = request.files["file"]
#
#     if file.filename == "":
#         return "No file selected"
#
#     if file:
#         # CSV 파일 열기
#         csv_reader = csv.reader(file.read().decode('utf-8').splitlines())
#
#         # 첫 줄은 헤더로 처리
#         header = next(csv_reader)
#
#         # 필드 인덱스 추출
#         temperature_idx = header.index("Temperature")
#         heartbeat_idx = header.index("Heartbeat")
#         gyrox_idx = header.index("GyroX")
#         gyroy_idx = header.index("GyroY")
#         gyroz_idx = header.index("GyroZ")
#
#         # 각 레코드를 한 줄씩 읽어 처리
#         for record in csv_reader:
#             # 레코드 처리
#             selected_data = [record[temperature_idx], record[heartbeat_idx], record[gyrox_idx], record[gyroy_idx], record[gyroz_idx]]
#             prediction = process_record(selected_data)
#             print(prediction)  # 예측 결과 출력
#
#             # 1초 대기
#             time.sleep(1)
#
#         return "Prediction completed"
#
# if __name__ == "__main__":
#     app.run(debug=True)

#=================================================================================================#
#=================================================================================================#
#=================================================================================================#
#=================================================================================================#

import time
import csv
from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# KNN 모델 경로 지정
knn_model_file = "./model/model_knn.pkl"
scaler_file = "./model/scaler_knn.pkl"

# KNN 모델 로드
knn_model = joblib.load(knn_model_file)

# Scaler 로드
scaler = joblib.load(scaler_file)

# 데이터 전처리 함수
def preprocess_data(data):
    scaled_data = scaler.transform(np.array(data).reshape(1, -1))
    return scaled_data

# CSV 파일 레코드 한 줄씩 처리 함수
def process_record(record):
    # 선택한 필드 추출
    selected_data = [float(record[0]), float(record[1]), float(record[2]), float(record[3]), float(record[4])]

    # 데이터 전처리
    preprocessed_data = preprocess_data(selected_data)

    # 예측
    prediction = knn_model.predict(preprocessed_data)

    # 예측 결과 반환
    return prediction.tolist()  # ndarray를 리스트로 변환하여 반환

# /predict 엔드포인트
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "파일이 제공되지 않았습니다."

    file = request.files["file"]

    if file.filename == "":
        return "선택된 파일이 없습니다."

    if file:
        # CSV 파일 열기
        csv_reader = csv.reader(file.read().decode('utf-8').splitlines())

        # 첫 줄은 헤더로 처리
        header = next(csv_reader)

        # 필드 인덱스 추출
        temperature_idx = header.index("Temperature")
        heartbeat_idx = header.index("Heartbeat")
        gyrox_idx = header.index("GyroX")
        gyroy_idx = header.index("GyroY")
        gyroz_idx = header.index("GyroZ")

        # 각 레코드를 한 줄씩 읽어 처리
        predictions = []
        for record in csv_reader:
            # 레코드 처리
            selected_data = [record[temperature_idx], record[heartbeat_idx], record[gyrox_idx], record[gyroy_idx], record[gyroz_idx]]
            prediction = process_record(selected_data)
            predictions.append(prediction)

            # 1초 대기
            time.sleep(1)

        return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)

