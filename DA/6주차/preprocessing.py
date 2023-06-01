import joblib

# 전처리 객체 불러오기
scaler = joblib.load('./model/scaler.pkl')

def preprocess_data(df):
    # 특성 이름 수정
    df = df.rename(columns={'gyroX': 'GyroX', 'gyroY': 'GyroY', 'gyroZ': 'GyroZ', 'heartbeat': 'Heartbeat', 'temperature': 'Temperature'})

    # 특성 순서 맞추기
    df = df[['Temperature', 'Heartbeat', 'GyroX', 'GyroY', 'GyroZ']]

    # 전처리
    X = scaler.transform(df)
    return X
