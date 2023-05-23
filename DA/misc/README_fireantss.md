# time-series-da

Data Analysis for Time Series Analysis

![Python v3.10.4](https://img.shields.io/badge/python-v3.10.4-3670A0?style=flat&logo=python&logoColor=ffdd54)
![pip v23.1.2](https://img.shields.io/badge/pip-v23.1.2-3670A0?style=flat&logo=python&logoColor=ffdd54)

- [가상환경구성하기](./docs/%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD%EA%B5%AC%EC%84%B1.md)
- [데이터분석 종합 문서(notebook)](./notebook/Data%20Analysis%20Comprehensive.ipynb)
- [Datasets](#dataset-보류)

## 참고자료

- [딥러닝을 이용한 비트코인 가격 예측 비교연구](./docs/A_Comparative_Study_of_Bitcoin_Price_Prediction_Us.pdf)

## Getting Started

1. 가상환경 실행

```bash
python -m venv venv
venv\Scripts\activate.bat
(venv) pip install -r requirements.txt
```

2. Run on Jupyterlab(Optional)

```bash
jupyter lab
# CMD: cntrl+Z // jupyterlab 종료
```

---

## TODO

- [ ] CNN(1D)

  - [x] .
    > .
  - [x] ~~Correlation Analysis~~

    > - 
    > - 

  - [ ] .

- .

- [ ] CNN(2D)
  - [ ] Conv2D .
  - [ ] Conv2D .
- [ ] LSTM
- [ ] CNN+LSTM
  - .
- [ ] Prophet
- [ ] GRU
- [ ] CNN + LSTM
- [ ] 데이터 분석 보고서 작성
- [ ] Flask

## Flask

- [A Comprehensive Guide on using Flask for Data Science](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-using-flask-for-data-science/)

- [Building a Machine Learning Web Application Using Flask](https://towardsdatascience.com/building-a-machine-learning-web-application-using-flask-29fa9ea11dac)

- [Deploying an ML web app using Flask](https://levelup.gitconnected.com/deploying-ml-web-app-using-flask-334367735777)

```bash
.
└──Flask-scikit
  ├── app.py
  ├── preprocessing.py
  ├──/templates
  │  ├── index.html
  │  └── result.html
  ├── /data
  ├── /notebook
  ├── /model
  │   ├── model.py
  │   ├── imputer.pkl
  │   ├── scaler.pkl
  │   └── model.pkl
  ├── /static
  │   └── style.css
  ├── requiremensts.txt
  └── README.md
```

> .

## .

> .

.

## .

.

## Memos

### .

- .
- .

#### Overfitting

.

![overfitting.png](./assets/overfitting.png)

.
![ideal.png](./assets/ideal.png)

### Optimizer

1. `adam`
2. .

## Issues

- [ ] API
      CNN-LSTM(kaggle버전)에서 사용하는 poloniex의 api는 `{period}` key(GET)로 불러올 수 있는 기간을 제한하고 있다. 더불어, 유일하게 제공하는 시계열 값인 timestamp가 unix timestamp이다.

이를 해결하기 위해 bitstamp에서 제공하는 1시간 단위의 dataset(`.csv`)을 이용한다.

단, 여전히 api로 실시간 데이터를 불러와 json을 csv로 저장하거나, 기존 데이터에 append할 수 있도록 준비하는 것은 중요하다.

- [x] ~~unix timestamps~~

  - 장점: 데이터 타입을 string이 아닌 long으로 받을 수 있다.
  - 단점: 인코딩 과정 필요(pd.to_datetime(..., unit='s')를 사용할 수 없는 경우 format으로 해결해야 한다.)

## Dataset _보류_

- data의 인덱스를 아래와 같이 고정.
- ~~dataset은 plato에 비공개~~

| Key       | Desc.      |
| :-------- | :--------- |
| timestamp | 날짜(시간) |
| open      | 시가       |
| close     | 종가       |
| high      | 고가       |
| low       | 저가       |
| vol       | 볼륨       |

- [투자자기준 10년 만기 미국채 수익률](./data/DGS10.csv)
- [GoldUSD Daily 19941206-20211206](./data/Gold_Daily.csv)
- [Stock-NewsEventsSentiments](./data/data.parquet)
- [BTCUSD Daily 20140917-20230504](./data/btc-usd-2014-2023.csv)
- [BTCUSD Daily 20120808-20230508](./data/2012to2023BTC-USD_investing.csv)
- [BTCUSD Minutely 20180714-20180826](./data/BTC-USD.csv): unixtimstamp
