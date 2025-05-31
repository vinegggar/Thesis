import os
import pandas as pd

def load_movielens(path):
    print(f"==> Шлях до даних: {os.path.abspath(path)}")
    try:
        entries = os.listdir(path)
    except Exception as e:
        print(f"Помилка читання директорії: {e}")
        raise
    print(f"==> Вміст теки:\n{entries}\n")

    if os.path.exists(os.path.join(path, "u.data")):  # 100k
        ratings = pd.read_csv(
            os.path.join(path, "u.data"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        movies = pd.read_csv(
            os.path.join(path, "u.item"),
            sep="|",
            encoding="latin-1",
            names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                   "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
                   "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
                   "Mystery","Romance","Sci-Fi","Thriller","War","Western"]
        )

        users = pd.read_csv(
            os.path.join(path, "u.user"),
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"]
        )

        movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

    elif os.path.exists(os.path.join(path, "ratings.dat")):  # 1m
        ratings = pd.read_csv(
            os.path.join(path, "ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        movies = pd.read_csv(
            os.path.join(path, "movies.dat"),
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding='latin-1'
        )

        users = pd.read_csv(
            os.path.join(path, "users.dat"),
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"]
        )

        for genre in sorted({g for gs in movies["genres"] for g in gs.split("|")}):
            movies[genre] = movies["genres"].str.contains(genre).astype(int)
        movies.drop(columns="genres", inplace=True)

        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

    else:
        raise ValueError("Unknown dataset format.")

    movies["year_norm"] = (movies["year"] - movies["year"].mean()) / movies["year"].std()

    return ratings, movies, users

