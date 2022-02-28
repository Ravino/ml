import pandas as pd


def task1():
    # 1.	Data Manipulation with pandas.

    # 1.1.	Создать датафрейм чтением данных из csv файла.

    # Пусть имеется датасет по фильмам и сериалам Netflix:
    df = pd.read_csv("./data/netflix_titles.csv")

    # 1.2.	Установить индекс созданного датафрейма.
    df = df.set_index("show_id")

    # 1.3.	Отсортировать датафрейм по значению какой-либо колонки.
    df = df.sort_values("title")

    # 1.4.	Сформировать выборку датафрейма по условию.
    df = df[df["release_year"] > 1980]

    # 1.5.	Сгруппировать выборку по набору из всех колонок,
    # кроме одной, оставшуюся колонку агрегировать любой функцией.
    group = df.groupby(["type", "rating", "country", "director",
                        "cast", "date_added", "title",
                        "duration", "listed_in", "description"])
    agg_func = {"release_year": ["mean"]}
    df = group.agg(agg_func)
    print(df)

    # 1.6.	Вывести информацию о полученном датафрейме.
    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    task1()
