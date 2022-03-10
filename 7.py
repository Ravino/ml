import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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
    distance_thresholds = [1, 10, 40, 45, 50, 60, 100]
    for distance_threshold in distance_thresholds:
        ac = AgglomerativeClustering(
            n_clusters=None,
            linkage="ward",
            distance_threshold=distance_threshold,
            compute_full_tree=True
        )
        ac_labels = ac.fit_predict(scaled_data)
        ac_score = silhouette_score(scaled_data, ac_labels)
        print(f"AgglomerativeClustering with ward linkage and "
              f"distance_threshold {distance_threshold} has a "
              f"silhouette score {ac_score:.3f}.")

    # Из полученных метрик можно сделать вывод о том, что для данной
    # задачи и модели AgglomerativeClustering оптимальнее всего
    # использовать distance_threshold=45, он дает метрику 0.154, тогда
    # как остальные значения дали метрику от 0.001 до 0.153.

    # 7.3.	Обучить модель кластеризации методом K-means и
    # оптимизировать ее гиперпараметры.

    # Аналогично подберем гиперпараметры для KMeans:
    random_seed = 10
    algorithms = ["full", "elkan"]
    inits = ["k-means++", "random"]
    n_clusters = list(range(2, 15))

    for algorithm in algorithms:
        for init in inits:
            for n_cluster in n_clusters:
                km = KMeans(n_clusters=n_cluster,
                            random_state=random_seed,
                            algorithm=algorithm,
                            init=init)
                km_labels = km.fit_predict(scaled_data)
                km_score = silhouette_score(scaled_data, km_labels)
                print(f"KMeans with algorithm {algorithm}, init {init}, "
                      f"and n_clusters {n_cluster} has a "
                      f"silhouette score {km_score:.3f}.")

    # По метрике silhouette для k means модели можно сделать выводы о
    # том, что оптимальнее всего использовать алгоритм elkan
    # со случайной инициализацией и числом кластеров 12, такой набор
    # дает метрику 0.160, что немного большей других конфигураций
    # (остальные от 0.1 до 0.152). Сравнивая с иерархической
    # кластеризацией можно наблюдать небольшой прирост значения метрики.


if __name__ == "__main__":
    task7()
