import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def task2():
    # 2.	Preprocessing for Machine Learning in Python.

    # 2.1.	Выбрать набор данных с числовыми, категориальными и
    # текстовыми колонками, отсутствующими значениями (pandas).

    # Пусть имеется датасет по пациентам, болеющим Covid-19 в Южной
    # Корее:
    df = pd.read_csv("./data/patient_info.csv")
    df.drop(columns=["country", "province", "city", "symptom_onset_date",
                     "confirmed_date", "released_date", "deceased_date",
                     "state", "infected_by"], inplace=True)

    # 2.2.	Удалить строки с отсутствующими значениями
    # категориальных и текстовых признаков (pandas).
    df = df.dropna(subset=["sex", "infection_case"])

    # 2.3.	Заполнить отсутствующие значения в числовых признаках
    # нулем (pandas).
    df["age"] = df["age"] \
        .fillna(0) \
        .apply(lambda age: float(age[:-1]) if ((age is not None)
                                               and (age != 0)) else 0)
    df["contact_number"] = df["contact_number"] \
        .fillna(0) \
        .apply(lambda n: float(n) if ((n is not None)
                                      and (n != "-")) else 0)

    # 2.4.	Преобразовать числовую колонку в строку и
    # обратно (pandas).
    df["age"] = df["age"].astype(str)
    df["age"] = df["age"].astype(float)

    # 2.5.	Нормализовать числовую колонку любым методом (sklearn).
    norm_scaler = MinMaxScaler()
    df["age"] = norm_scaler.fit_transform(df["age"].values.reshape(-1, 1))

    # 2.6.	Стандартизовать другую числовую колонку (sklearn).
    std_scaler = StandardScaler()
    df["contact_number"] = std_scaler.fit_transform(df["contact_number"]
                                                    .values.reshape(-1, 1))

    # 2.7.	Применить one-hot encoding к колонке с категориальными
    # значениями (pandas/sklearn).
    encoder = OneHotEncoder()
    df["sex"] = encoder.fit_transform(df["sex"]
                                      .values.reshape(-1, 1)).toarray()

    # 2.8.	Векторизовать текстовую колонку методом tf-idf (sklearn).
    vectorizer = TfidfVectorizer()
    df["infection_case"] = vectorizer.fit_transform(df["infection_case"]
                                                    .values).toarray()

    # 2.9.	Удалить избыточные колонки на основе анализа корреляции
    # (pandas).
    corr = df.corr()
    print(corr)
    # Матрица корреляции говорит о том, что наибольшую корреляцию имеют
    # столбцы "patient_id" и "age". Скорее всего, это случайность, так
    # как индекс не может коррелировать с возрастом пациента для
    # данного датасета. Пусть для выполнения задания избыточной
    # колонкой будет "patient_id".
    df.drop(columns=["patient_id"], inplace=True)

    # 2.10.	Сократить размерность набора данных с помощью метода
    # главных компонент (sklearn).
    pca = PCA(n_components=2)
    pca_array = pca.fit_transform(df)
    print(pca_array)


if __name__ == "__main__":
    task2()
