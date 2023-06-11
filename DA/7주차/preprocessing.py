# def preprocess_data(df):
#     # 특성 이름 수정
#     df = df.rename(columns={'gyroX': 'gx', 'gyroY': 'gy', 'gyroZ': 'gz'})
#
#     # 특성 순서 맞추기
#     df = df[['gx', 'gy', 'gz']]
#
#     return df

import numpy as np

def preprocess_data(df):
    # 특성 이름 수정
    df = df.rename(columns={'gyroX': 'gx', 'gyroY': 'gy', 'gyroZ': 'gz'})

    # 특성 순서 맞추기
    df = df[['gx', 'gy', 'gz']]

    # 40줄의 레코드를 하나의 시퀀스로 묶기
    num_records = len(df)
    num_sequences = num_records // 40
    sequences = []

    for i in range(num_sequences):
        start_idx = i * 40
        end_idx = start_idx + 40
        sequence = df[start_idx:end_idx].values
        sequences.append(sequence)

    return np.array(sequences)
