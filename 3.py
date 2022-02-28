from datetime import datetime

import numpy as np
from hyperopt import hp, fmin, tpe
from lightgbm.sklearn import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, \
    GridSearchCV, RandomizedSearchCV
from tpot import TPOTRegressor


def task3():
    # 3.	Hyperparameter Tuning in Python

    # 3.1.	Выбрать задачу машинного обучения, набор данных,
    # алгоритм.

    # Пусть необходимо решить задачу регрессии на основе датасета из
    # библиотеки sklearn по пациентам для предсказания количественной
    # оценки заболевания через год после исходного уровня:
    dataset = load_diabetes()
    data = dataset.data
    target = dataset.target

    random_seed = 9
    np.random.seed(random_seed)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=random_seed)

    model = LGBMRegressor(random_state=random_seed)
    scoring = "neg_mean_squared_error"
    score = -cross_val_score(model,
                             train_data,
                             train_targets,
                             scoring=scoring).mean()
    print(f"default score is {score}.")

    # 3.2.	Оптимизировать гиперпараметры модели поиском по
    # решетке (sklearn).
    parameters = {
        "max_depth": np.linspace(2, 20, 9, dtype=int),
        "n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "random_state": [random_seed],
    }
    grid_search = GridSearchCV(model,
                               parameters,
                               scoring=scoring)

    grid_t_0 = datetime.now()
    grid_search.fit(train_data, train_targets)
    grid_dt = datetime.now() - grid_t_0
    grid_score = mean_squared_error(test_targets,
                                    grid_search.predict(test_data))
    print(f"grid search took {grid_dt.seconds} s with MSE {grid_score} "
          f"with parameters {grid_search.best_params_}.")

    # 3.3.	Оптимизировать гиперпараметры модели случайным поиском
    # (sklearn).
    random_search = RandomizedSearchCV(model,
                                       parameters,
                                       scoring=scoring,
                                       random_state=random_seed)

    random_t_0 = datetime.now()
    random_search.fit(train_data, train_targets)
    random_dt = datetime.now() - random_t_0
    random_score = mean_squared_error(test_targets,
                                      random_search.predict(test_data))
    print(f"random search took {random_dt.seconds} s with MSE {random_score} "
          f"with parameters {random_search.best_params_}.")

    # 3.4.	Оптимизировать гиперпараметры модели с помощью
    # байесовской оптимизации (Hyperopt).
    hp_parameters = {
        "max_depth": hp.quniform("max_depth", 2, 20, 2),
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 100)
    }

    def f(params):
        hp_model = LGBMRegressor(random_state=random_seed,
                                 max_depth=int(params["max_depth"]),
                                 n_estimators=int(params["n_estimators"])
                                 )
        f_score = -cross_val_score(hp_model,
                                   train_data,
                                   train_targets,
                                   scoring=scoring).mean()
        return f_score

    hp_t_0 = datetime.now()
    hp_best_params = fmin(
        fn=f,
        space=hp_parameters,
        algo=tpe.suggest,
        max_evals=50
    )
    hp_dt = datetime.now() - hp_t_0
    model = LGBMRegressor(random_state=random_seed,
                          max_depth=int(hp_best_params["max_depth"]),
                          n_estimators=int(hp_best_params["n_estimators"]))
    model.fit(train_data, train_targets)
    hp_score = mean_squared_error(test_targets, model.predict(test_data))
    print(f"hyperopt took {hp_dt.seconds} s with MSE {hp_score} "
          f"with parameters {hp_best_params}.")

    # 3.5.	Оптимизировать гиперпараметры модели с помощью
    # генетического программирования (TPOT).
    tpot_t_0 = datetime.now()
    tpot_regressor = TPOTRegressor(
        generations=3,
        population_size=10,
        verbosity=2,
        config_dict={"lightgbm.sklearn.LGBMRegressor": parameters},
        scoring=scoring,
        random_state=random_seed)
    tpot_regressor.fit(train_data, train_targets)
    tpot_dt = datetime.now() - tpot_t_0
    tpot_score = -tpot_regressor.score(test_data, test_targets)
    print(f"tpot took {tpot_dt.seconds} s with MSE {tpot_score}.")

    # 3.6.	Сравнить методы оптимизации по времени и точности модели.

    # Приведем следующий вывод из терминала для данной задачи:
    # default score is 3980.121650624278.
    # grid search took 35 s with MSE 2769.8453546019296 with parameters
    # {'max_depth': 2, 'n_estimators': 100, 'random_state': 9}.
    # random search took 4 s with MSE 3108.340329433587 with
    # parameters {'random_state': 9, 'n_estimators': 400, 'max_depth': 2}.
    # hyperopt took 16 s with MSE 2769.8453546019296 with parameters
    # {'max_depth': 2.0, 'n_estimators': 100.0}.
    # Best pipeline: LGBMRegressor(CombineDFs(input_matrix, input_matrix),
    # max_depth=2, n_estimators=100, random_state=9)
    # tpot took 20 s with MSE 2769.8453546019296.

    # Исходя из результатов можно сделать следующие выводы:
    # 1. Среднеквадратичная ошибка для модели, обученной на параметрах
    # по умолчанию, является самой большой по сравнению с ошибками на
    # тех моделях, гиперпараметры которых были оптимизированы. Это
    # говорит о том, что есть смысл оптимизировать гиперпараметры для
    # конкретной постановки задачи.
    # 2. Поиск по решетке уступает по времени случайному поиску и другим
    # методам оптимизации, однако дает более оптимальные параметры.
    # 3. Байесовскуб оптимизацию имеет смысл применять потому, что она
    # нашла самые оптимальные параметры за время, меньшее поиска по
    # сетке в 2 раза.
    # 4. Генетические алгоритмы в данной задаче показали свое
    # преимущество перед поиском по решетке с той же ошибкой, однако
    # уступили более быстрому методу байесовской оптимизации.
    # 5. В итоге, оптимальнее с точки зрения скорости и точности
    # в данной задаче регрессии оказалась библиотека hyperopt с ее
    # методом поиска оптимальных гиперпараметров модели.


if __name__ == "__main__":
    task3()
