import numpy as np
import pandas
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def task8():
    # 8.	Introduction to Deep Learning with Keras.

    # 8.1.	Выбрать набор данных для бинарной классификации.

    random_seed = 22
    np.random.seed(random_seed)

    # В данном задании пусть есть данные по камням и металлу, на которые
    # воздействовали сонаром и получили различные характеристики. По
    # ним нужно сделать решение, металл ли это, или камень.
    dataframe = pandas.read_csv("data/sonar.csv", header=None)
    dataset = dataframe.values
    data = dataset[:, 0:60].astype(float)
    target = dataset[:, 60]

    # Чтобы использовать метки, их нужно преобразовать в числа:
    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    # Также можно стандартизовать данные:
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=random_seed)

    # 8.2.	Обучить многослойный перцептрон для бинарной классификации.
    units = 10

    model1 = Sequential()
    model1.add(Dense(units))
    model1.add(Dense(1, activation="sigmoid"))
    model1.compile(loss="binary_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    model1.fit(train_data,
               train_targets,
               validation_data=(test_data, test_targets),
               epochs=1000,
               verbose=0)
    _, train_acc = model1.evaluate(train_data, train_targets, verbose=0)
    _, test_acc = model1.evaluate(test_data, test_targets, verbose=0)
    print(f"train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")

    # Видим, что аккуратность достигла единицы для обучающей выборки,
    # при этом на тестовой выборке эта метрика значительно меньше.
    # Это происходит потому, что модель переобучилась. В такие моменты
    # функция потерь для тестовой выборки перестает уменьшаться и
    # начинает увеличиваться. Чтобы этого избежать, необходимо
    # прекращать обучение в этой точке.

    # 8.3.	Обучить многослойный перцептрон с ранней остановкой.
    patience = 50

    model2 = Sequential()
    model2.add(Dense(units))
    model2.add(Dense(1, activation="sigmoid"))
    model2.compile(loss="binary_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss",
                       verbose=1,
                       patience=patience)
    model2.fit(train_data,
               train_targets,
               validation_data=(test_data, test_targets),
               epochs=1000,
               verbose=0,
               callbacks=[es])
    _, train_acc = model2.evaluate(train_data, train_targets, verbose=0)
    _, test_acc = model2.evaluate(test_data, test_targets, verbose=0)
    print(f"train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")

    # С ранней остановкой получилось достигнуть большую аккуратность на
    # тестовой выборке, чем без нее. Также получилось выполнить меньшее
    # число шагов обучения, чем без использования ранней остановки.

    # 8.4.	Обучить многослойный перцептрон с разными функциями
    # активации.
    acts = ["linear", "relu", "elu", "tanh"]
    for act in acts:
        model3 = Sequential()
        model3.add(Dense(units, activation=act))
        model3.add(Dense(1, activation="sigmoid"))
        model3.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss",
                           verbose=1,
                           patience=patience)
        model3.fit(train_data,
                   train_targets,
                   validation_data=(test_data, test_targets),
                   epochs=1000,
                   verbose=0,
                   callbacks=[es])
        _, train_acc = model3.evaluate(train_data,
                                       train_targets,
                                       verbose=0)
        _, test_acc = model3.evaluate(test_data,
                                      test_targets,
                                      verbose=0)
        print(f"model with {act} activation in the hidden layer"
              f" - train accuracy: {train_acc:.3f}, "
              f"test accuracy: {test_acc:.3f}")

    # По результатам можно сделать выводы о том, что оптимальнее всего
    # использовать скрытые слои с функциями активации ReLu. Однако,
    # комбинации из другий функций активации также имеют место быть для
    # данной задачи, так как дает схожие аккуратности.

    # 8.5.	Обучить многослойный перцептрон с нормализацией батчей.
    model4 = Sequential()
    model4.add(Dense(units))
    model4.add(BatchNormalization())
    model4.add(Activation("tanh"))
    model4.add(Dense(1, activation="sigmoid"))
    model4.compile(loss="binary_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss",
                       verbose=1,
                       patience=patience)
    model4.fit(train_data,
               train_targets,
               validation_data=(test_data, test_targets),
               epochs=1000,
               verbose=0,
               callbacks=[es])
    _, train_acc = model4.evaluate(train_data, train_targets, verbose=0)
    _, test_acc = model4.evaluate(test_data, test_targets, verbose=0)
    print(f"model with tanh activation in the hidden layer and batch "
          f"normalization - train accuracy: {train_acc:.3f}, "
          f"test accuracy: {test_acc:.3f}")

    # Как можно видеть, батч нормализация позволила несколько уменьшить
    # число эпох обучения в сравнении с моделью без нормализации.
    # В такой небольшой модели это может быть незаметно, но в больших
    # моделях это может сыграть значительную роль при обучении модели.

    # 8.6.	Сравнить точность обученных сетей.

    # Из полученных результатов и сделанных выше выводов можно
    # заключить, что в данной задаче бинарной классификации оптимально
    # использовать методы, применяемые при обучении нейронных сетей,
    # такие как: ранняя остановка для избавления от переобучения,
    # выбор подходящей функции активации скрытого слоя многослойного
    # перцептрона, а также использование батч-нормализации.


if __name__ == "__main__":
    task8()
