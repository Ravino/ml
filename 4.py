import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD


def task4():
    # 4.	Building Recommendation Engines in Python.

    # 4.1.	Выбрать набор данных с контентом (товары, фильмы,
    # книги и т.д.).

    # Пусть в качестве датасета для создания модели рекомандации на
    # основе сходства контента выступит набор данных по товарам из
    # спортивного магазина. В датасете есть столбец номера товара, а
    # также информация о нем в формате имя - описание.
    df = pd.read_csv("data/content_based.csv")

    # 4.2.	Разработать модель, формирующую рекомендации на основе
    # анализа сходства контента.

    # Векторизуем описание методом tf-idf:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["description"])

    # Строим матрицу косинусных коэффициентов между описаниями товаров:
    cs = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Теперь на основе этой матрицы можно найти схожие товары.
    # Пусть стоит задача для товара с индексом item_index найти
    # num_recommend схожих товаров.
    item_index = 11
    num_recommend = 5
    similar_indices = np.argsort(cs[item_index - 1])
    # Пропустим самый похожий товар -- самого себя, который обязательно
    # включен в список:
    similar_indices = similar_indices[::-1][1:(num_recommend + 1)]
    similar_cs = cs[item_index - 1][similar_indices]
    current_item = df[df["id"] == item_index]["description"]
    current_item = current_item.values[0].split(" - ")[0]

    # Выведем информацию о полученных рекомендациях:
    print(f"{num_recommend} similar items to \"{current_item}\" are:")
    for i in range(num_recommend):
        recommended_item = df[df["id"] == (similar_indices[i] + 1)]
        recommended_item = recommended_item["description"]
        recommended_item = recommended_item.values[0].split(" - ")[0]
        print(f"\"{recommended_item}\" recommended with cosine "
              f"similarity {similar_cs[i]}")
    # Названия рекомендованных товаров схожи с тем, который был
    # нами выбран.

    # 4.3.	Выбрать набор данных с пользователями (покупки,
    # предпочтения, просмотренные фильмы).

    # Пусть теперь имеются данные по фильмам и пользователям сервиса.
    # Здесь имеются id пользователей, id фильма, и рейтинг, присвоенный
    # этому фильму данным пользователем.
    ratings = pd.read_csv("data/ratings_small.csv")
    ratings.drop(columns=["timestamp"], inplace=True)

    # 4.4.	Разработать модель, формирующую рекомендации на основе
    # сходства с другими пользователями (например, KNN).
    reader = Reader()
    data = Dataset.load_from_df(ratings, reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    # Теперь пусть нужно порекомендовать пользователю uid num_recommend
    # фильмов:
    uid = 1
    num_recommend = 5
    movie_ids = set(ratings["movieId"].unique())
    user_movie_ids = set(ratings[ratings['userId'] == uid]["movieId"].unique())
    not_watched_movie_ids = np.array(list(movie_ids - user_movie_ids))
    predicted = np.array([svd.predict(uid, i).est
                          for i in not_watched_movie_ids])
    arg_ids = predicted.argsort()[::-1][:num_recommend]
    top_recommended_movie_ids = not_watched_movie_ids[arg_ids]

    # Выведем информацию о полученных рекомендациях:
    print(f"{num_recommend} recommended films for user \"{uid}\" are:")
    for i in range(num_recommend):
        print(f"\"{top_recommended_movie_ids[i]}\" is recommended with "
              f"predicted rating {predicted[arg_ids][i]}")

    # Таким образом, если стоит задача порекомендовать пользователю
    # несколько фильмов, то имеет смысл порекомендовать такие, которые
    # пользователь скорее всего оценит лучше всего.


if __name__ == "__main__":
    task4()
