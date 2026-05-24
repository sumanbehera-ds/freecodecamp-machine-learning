import pandas as pd
from sklearn.neighbors import NearestNeighbors


def get_recommends(book=""):
    books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False)
    ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip")

    ratings_count = ratings["ISBN"].value_counts()
    users_count = ratings["User-ID"].value_counts()

    ratings = ratings[
        ratings["ISBN"].isin(ratings_count[ratings_count >= 100].index)
        & ratings["User-ID"].isin(users_count[users_count >= 200].index)
    ]

    merged = ratings.merge(books, on="ISBN")

    book_matrix = merged.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(book_matrix)

    if book not in book_matrix.index:
        return [book, []]

    distances, indices = model.kneighbors(
        [book_matrix.loc[book]],
        n_neighbors=6
    )

    recommended_books = []

    for i in range(5, 0, -1):
        title = book_matrix.index[indices[0][i]]
        distance = distances[0][i]
        recommended_books.append([title, distance])

    return [book, recommended_books]