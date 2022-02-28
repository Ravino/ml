import numpy as np
import pandas as pd
import xgboost as xgb
from keras.losses import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg


def task5():
    # 5.	Machine Learning for Time Series Data in Python.

    # 5.1.	Выбрать набор данных с временным рядом и числовой колонкой.

    # Допустим, есть данные по числу автомобилей, которые проехали за
    # день через Baregg Tunnel в Швейцарии в 2003-2005 годах:
    df = pd.read_csv("data/tunnel.csv", parse_dates=["Day"])

    # 5.2.	Установить временной ряд в качестве индекса набора данных.
    df.set_index("Day", inplace=True)

    # Так как данные имеют период 1 день, имеет смысл создать колонку
    # с индексом, которая будет использована для обучения моделей:
    df["Time"] = np.arange(len(df.index))

    # 5.3.	Создать новую колонку со смещенными во времени
    # значениями числовой колонки.

    df["ShiftedNumVehicles"] = df["NumVehicles"].shift(periods=1,
                                                       fill_value=-1)

    # Также разделим датасет для обучения и тестирования:
    split_date = "01-June-2005"
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    # 5.4.	Обучить регрессионную модель на основе исходной и
    # смещенной колонок.

    # Рассмотрим две модели: линейную регрессию на основе исходной и
    # смещенной колонок и градиентный бустинг на основе преобразованных
    # признаков.

    data_train = df_train["ShiftedNumVehicles"].to_numpy().reshape(-1, 1)
    target_train = df_train["NumVehicles"].to_numpy().reshape(-1, 1)
    data_test = df_test["ShiftedNumVehicles"].to_numpy().reshape(-1, 1)
    target_test = df_test["NumVehicles"]

    lr = LinearRegression()
    lr.fit(data_train, target_train)
    lr_target_predicted = lr.predict(data_test).reshape(1, -1)

    mape_lr = mean_absolute_percentage_error(target_test,
                                             lr_target_predicted)
    print(f"MAPE for LinearRegression is {mape_lr[0]:.2f}%.")
    # Таким образом, метрика MAPE примерно в 9% говорит о том, что
    # модель линейной регрессии может связывать значения числа
    # автомобилей с этим числом за день до события, то есть показывает
    # связь смещенной и несмещенной колонок.

    # Дополнительно построим более сложную модель прогнозирования.

    # На основе даты можно сформировать новые признаки для данных для
    # обучения:
    def create_features(dataframe, label=None):
        dataframe["date"] = dataframe.index
        dataframe["dayofweek"] = dataframe["date"].dt.dayofweek
        dataframe["quarter"] = dataframe["date"].dt.quarter
        dataframe["month"] = dataframe["date"].dt.month
        dataframe["year"] = dataframe["date"].dt.year
        dataframe["dayofyear"] = dataframe["date"].dt.dayofyear
        dataframe["dayofmonth"] = dataframe["date"].dt.day
        dataframe["weekofyear"] = dataframe["date"].dt.weekofyear

        data = dataframe[["dayofweek",
                          "quarter",
                          "month",
                          "year",
                          "dayofyear",
                          "dayofmonth",
                          "weekofyear"]]
        if label:
            target = dataframe[label]
            return data, target

        return data

    data_train_full, target_train_full = create_features(df_train,
                                                         label="NumVehicles")
    data_test_full, target_test_full = create_features(df_test,
                                                       label="NumVehicles")

    # Пусть в качестве регрессионной модели выступит градиентный бустинг
    # из библиотеки xgboost:
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(data_train_full, target_train_full,
            eval_set=[(data_train_full, target_train_full),
                      (data_test_full, target_test_full)],
            early_stopping_rounds=50,
            verbose=0)

    target_predicted = reg.predict(data_test_full)

    # Посмотрим на метрику MAPE:
    mape_xgb_reg = mean_absolute_percentage_error(target_test_full,
                                                  target_predicted)
    print(f"MAPE for XGBRegressor is {mape_xgb_reg:.2f}%.")
    # Ошибка MAPE для XGBRegressor составила около 3%.
    # В последнем случае мы решили задачу прогнозирования числа
    # автомобилей в конкретный день.

    # 5.5.	Обучить авторегрессионную модель на основе исходной
    # колонки.

    train_ar = df_train["NumVehicles"]
    test_ar = df_test["NumVehicles"]

    ar = AutoReg(train_ar, lags=50)
    model_fit = ar.fit()
    predictions = model_fit.predict(start=len(train_ar),
                                    end=len(train_ar)+len(test_ar)-1,
                                    dynamic=False)
    mape_ar = mean_absolute_percentage_error(test_ar, predictions)
    print(f"MAPE for AutoReg is {mape_ar:.2f}%.")
    # Как видим, результат для авторегрессионной модели лучше, чем при
    # использовании лишь одной смещенной колонки, как было показано
    # в пункте с линейной регрессии. Видимо, это потому, что применяется
    # большее число временной информации, что дает больше информации для
    # обучения модели и, вследствие, приводит к увеличению ее точности.
    # Однако, эта метрика все равно хуже, чем при использовании
    # градиентного бустинга с созданием дополнительных признаков, что
    # говорит о полезности инжиниринга исходных фич.


if __name__ == "__main__":
    task5()
