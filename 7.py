import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def task7():
    # 7.	Cluster Analysis in Python.

    # 7.1.	Выбрать набор данных для кластеризации/классификации.

    # Теперь пусть данные представляют из себя изображения рукописных
    # цифр от 0 до 9. Таким образом нужно кластеризовать эти изображения
    # так, чтобы отделить похожие.

    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(f"# of digits: {n_digits}; "
          f"# of samples: {n_samples}; "
          f"# of features {n_features}")

    # Чтобы использовать эти данные, их нужно стандартизовать:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 7.2.	Обучить модель иерархической кластеризации и
    # оптимизировать ее гиперпараметры.

    # Для упрощения подбор гиперпараметров выполнен вручную перебором:
    linkages = ["ward", "complete", "average", "single"]
    for linkage in linkages:
        ac = AgglomerativeClustering(n_clusters=n_digits, linkage=linkage)
        ac_labels = ac.fit_predict(scaled_data)
        ac_score = silhouette_score(scaled_data, ac_labels)
        print(f"AgglomerativeClustering with {linkage} linkage has a "
              f"silhouette score {ac_score:.3f}.")

    # Из полученных метрик можно сделать вывод о том, что для данной
    # задачи и модели AgglomerativeClustering оптимальнее всего
    # использовать average linkage.

    # 7.3.	Обучить модель кластеризации методом K-means и
    # оптимизировать ее гиперпараметры.

    # Аналогично подберем гиперпараметры для KMeans:
    random_seed = 10
    algorithms = ["full", "elkan"]
    inits = ["k-means++", "random"]

    for algorithm in algorithms:
        for init in inits:
            km = KMeans(n_clusters=n_digits,
                        random_state=random_seed,
                        algorithm=algorithm,
                        init=init)
            km_labels = km.fit_predict(scaled_data)
            km_score = silhouette_score(scaled_data, km_labels)
            print(f"KMeans with algorithm {algorithm} and init {init} has a "
                  f"silhouette score {km_score:.3f}.")

    # По метрике silhouette для k means модели можно сделать выводы о
    # том, что можно использовать любой из реализованных алгоритмов в
    # sklearn, а метод инициализации -- k-means++. При этом можно
    # заметить, что эта метрика заметно меньше (0.147),
    # чем при использовании иерархической кластеризации (0.532).


if __name__ == "__main__":
    task7()
