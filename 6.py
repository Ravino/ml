from datetime import datetime

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def task6():
    # 6.	Выбрать набор данных для классификации.

    # Пусть нужно решить задачу мультиклассификации вина на основе
    # канонического набора библиотеки sklearn.
    dataset = load_wine()
    data = dataset.data
    target = dataset.target

    random_seed = 10
    np.random.seed(random_seed)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=random_seed)

    # Пусть f1-мера будет рассчитываться как взвешенное среднее:
    average = "weighted"

    # 6.1.	Обучить классификатор на основе бэггинга.
    b_classifier = BaggingClassifier()
    b_classifier.fit(train_data, train_targets)
    bc_predicted = b_classifier.predict(test_data)
    bc_score = f1_score(test_targets, bc_predicted, average=average)
    print(bc_score)

    # 6.2.	Обучить классификатор AdaBoost.
    ab_classifier = AdaBoostClassifier()
    ab_classifier.fit(train_data, train_targets)
    ab_predicted = ab_classifier.predict(test_data)
    ab_score = f1_score(test_targets, ab_predicted, average=average)
    print(ab_score)

    # 6.3.	Обучить классификатор на основе градиентного бустинга.
    gb_classifier = GradientBoostingClassifier()
    gb_classifier.fit(train_data, train_targets)
    gb_predicted = gb_classifier.predict(test_data)
    gb_score = f1_score(test_targets, gb_predicted, average=average)
    print(gb_score)

    # 6.4.	Оптимизировать гиперпараметры моделей.
    # Подбор гиперпараметров для бэггинга:
    bc_parameters = {
        "n_estimators": [50, 100, 500, 1000],
        "max_samples": [0.1, 0.2, 0.5],
        "random_state": [random_seed],
    }
    bc_t_0 = datetime.now()
    bc_grid_search = GridSearchCV(BaggingClassifier(),
                                  bc_parameters,
                                  scoring="f1_" + average)
    bc_grid_search.fit(train_data, train_targets)
    bc_dt = datetime.now() - bc_t_0
    bc_grid_score = f1_score(test_targets,
                             bc_grid_search.predict(test_data),
                             average=average)
    print(f"f1 for BaggingClassifier is {bc_grid_score}, "
          f"took {bc_dt.seconds} sec to search.")

    # Подбор гиперпараметров для AdaBoost:
    ab_parameters = {
        "n_estimators": [50, 100, 500, 1000],
        "learning_rate": [0.1, 0.5, 1],
        "random_state": [random_seed],
    }
    ab_t_0 = datetime.now()
    ab_grid_search = GridSearchCV(AdaBoostClassifier(),
                                  ab_parameters,
                                  scoring="f1_" + average)
    ab_grid_search.fit(train_data, train_targets)
    ab_dt = datetime.now() - ab_t_0
    ab_grid_score = f1_score(test_targets,
                             ab_grid_search.predict(test_data),
                             average=average)
    print(f"f1 for AdaBoostClassifier is {ab_grid_score}, "
          f"took {ab_dt.seconds} sec to search.")

    # Подбор гиперпараметров для градиентного бустинга:
    gb_parameters = {
        "n_estimators": [50, 100, 500, 1000],
        "learning_rate": [0.1, 0.5, 1],
        "random_state": [random_seed],
    }
    gb_t_0 = datetime.now()
    gb_grid_search = GridSearchCV(GradientBoostingClassifier(),
                                  gb_parameters,
                                  scoring="f1_" + average)
    gb_grid_search.fit(train_data, train_targets)
    gb_dt = datetime.now() - gb_t_0
    gb_grid_score = f1_score(test_targets,
                             gb_grid_search.predict(test_data),
                             average=average)
    print(f"f1 for GradientBoostingClassifier is {gb_grid_score}, "
          f"took {gb_dt.seconds} sec to search.")

    # 6.5.	Сравнить модели по точности.

    # По полученным значениям f1-меры для обученных моделей с
    # параметрами, заданными по-умолчанию, можно сказать, что лучше
    # всего показала себя модель бэггинга. Однако подбор гиперпараметров
    # помог получить более высокие по сравнению с другими моделями
    # метрики для алгоритма AdaBoost. По времени подбора гиперпараметров
    # также выигрывает классификатор AdaBoost. Вероятно, можно получить
    # более точную модель градиентного бустинга, если рассматривать
    # большее число гиперпараметров.

    # 6.6.	Вывести feature importance лучшей модели.
    best_model = AdaBoostClassifier(**ab_grid_search.best_params_)
    best_model.fit(train_data, train_targets)
    bm_score = f1_score(test_targets,
                        best_model.predict(test_data),
                        average=average)
    print(f"best model (AdaBoost) f1 score is {bm_score}.")

    r = permutation_importance(best_model,
                               test_data,
                               test_targets,
                               n_repeats=30,
                               random_state=random_seed)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{dataset.feature_names[i]} "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

    # Полученные значение feature importance позволяют сказать о том,
    # что наиболее важными признаками для модели являются:
    # - flavanoids;
    # - color_intensity;
    # - alcohol.


if __name__ == "__main__":
    task6()
