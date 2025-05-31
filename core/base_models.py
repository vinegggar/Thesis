import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from surprise import SVD, Dataset, Reader

class Recommender(ABC):
    """
    Base class
    """
    def fit(self, train_df):
        """
        Train the model
        """
        pass

    @abstractmethod
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for given user–item pairs.
        Returns a list/array of predictions: f(u, i) = r̂_ui
        """
        pass

    def evaluate(self, test_df):
        """
        Evaluate model using RMSE and R squared
        """
        user_ids = test_df["user_id"].values
        item_ids = test_df["movie_id"].values
        true_ratings = test_df["rating"].values
        preds = self.predict(user_ids, item_ids)

        rmse = root_mean_squared_error(true_ratings, preds)
        r2 = r2_score(true_ratings, preds)

        return preds, true_ratings,  {"rmse": rmse, "r2": r2}


class UserCFRecommender(Recommender):
    """
    User-based Collaborative Filtering
    """

    def __init__(self, default_rating=3.53, eps=1e-12, k_neighbors=None):
        self.default_rating = default_rating
        self.eps = eps
        self.k_neighbors = k_neighbors

    def fit(self, train_df):
        self.R = train_df.pivot(index='user_id', columns='movie_id', values='rating')
        self.user_means = self.R.mean(axis=1)
        R_centered = self.R.sub(self.user_means, axis=0).fillna(0)
        sim = cosine_similarity(R_centered)
        self.sim_df = pd.DataFrame(sim, index=self.R.index, columns=self.R.index)

    def predict(self, user_ids, item_ids):
        preds = []
        k = self.k_neighbors

        for u, i in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Predicting w User-CF"):
            if u not in self.R.index or i not in self.R.columns:
                preds.append(self.default_rating)
                continue

            ratings_i = self.R[i]
            neighbors = ratings_i.dropna().index  # users who rated item i

            if len(neighbors) == 0:
                preds.append(self.user_means.get(u, self.default_rating))
                continue

            sims = self.sim_df.loc[u, neighbors]
            neighbor_ratings = ratings_i.loc[neighbors]
            neighbor_means = self.user_means.loc[neighbors]
            deviations = neighbor_ratings - neighbor_means

            if k and len(sims) > k:
                top_k_neighbors = sims.abs().nlargest(k).index
                sims = sims.loc[top_k_neighbors]
                deviations = deviations.loc[top_k_neighbors]

            numerator = np.dot(sims.values, deviations.values)
            denominator = np.abs(sims.values).sum() + self.eps
            base = self.user_means.get(u, self.default_rating)
            pred = base + numerator / denominator

            preds.append(pred)

        return  np.array(preds)

class ItemCFRecommender(Recommender):
    """
    Item-based Collaborative Filtering
    """
    def __init__(self, default_rating=3.53, eps=1e-12, k_neighbors=None):
        self.default_rating = default_rating
        self.eps = eps
        self.k_neighbors = k_neighbors

    def fit(self, train_df):
        self.R = train_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        sim = cosine_similarity(self.R.T)
        self.sim_df = pd.DataFrame(sim, index=self.R.columns, columns=self.R.columns)

    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Predicting w Item-CF"):
            if u not in self.R.index or i not in self.R.columns:
                preds.append(self.default_rating)
                continue

            user_ratings = self.R.loc[u]
            rated_items = user_ratings[user_ratings > 0].index

            if len(rated_items) == 0:
                preds.append(self.item_means.get(i, self.default_rating))
                continue

            sims = self.sim_df.loc[i, rated_items]
            neighbor_ratings = user_ratings.loc[rated_items]

            # Top-k most similar items the user has rated
            if self.k_neighbors and len(sims) > self.k_neighbors:
                top_k = sims.abs().nlargest(self.k_neighbors).index
                sims = sims.loc[top_k]
                neighbor_ratings = neighbor_ratings.loc[top_k]

            numerator = np.dot(sims.values, neighbor_ratings.values)
            denominator = np.abs(sims.values).sum() + self.eps
            pred = numerator / denominator
            preds.append(pred)

        return  np.array(preds)

class CBFRecommender(Recommender):
    """
    Content-Based Filtering Recommender using item metadata (genres, year).
    """

    def __init__(self, default_rating=3.53, eps=1e-12, k_neighbors=None):
            self.default_rating = default_rating
            self.eps = eps
            self.k_neighbors = k_neighbors

    def fit(self, train_df, items_df, feature_cols):
        self.user_ratings = train_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        self.user_means = (
            self.user_ratings
            .replace(0, np.nan)
            .mean(axis=1)
            .fillna(self.default_rating)
        )

        self.item_profiles = (
            items_df
            .set_index("movie_id")[feature_cols]
            .fillna(0)
        )

        sim = cosine_similarity(self.item_profiles.values)
        self.item_sim = pd.DataFrame(
            sim,
            index=self.item_profiles.index,
            columns=self.item_profiles.index
        )

    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Predicting w CBF"):
            if u not in self.user_ratings.index or i not in self.item_sim.index:
                preds.append(self.default_rating)
                continue

            ratings_u = self.user_ratings.loc[u]
            rated = ratings_u[ratings_u > 0]
            if rated.empty:
                preds.append(self.user_means.get(u, self.default_rating))
                continue

            sims = self.item_sim.loc[i, rated.index]
            deviations = rated - self.user_means[u]

            if self.k_neighbors and len(sims) > self.k_neighbors:
                topk_idx = sims.abs().nlargest(self.k_neighbors).index
                sims = sims.loc[topk_idx]
                deviations = deviations.loc[topk_idx]

            num = np.dot(sims.values, deviations.values)
            den = np.abs(sims.values).sum() + self.eps
            pred = self.user_means[u] + num / den
            preds.append(pred)

        return  np.array(preds)

class HybridRecommender(Recommender):
    """
    Weighted combination of two recommenders
    """
    def __init__(self, model1, model2, blend=0.5):
        self.model1 = model1
        self.model2 = model2
        self.blend = blend

    def predict(self, user_ids, item_ids):
        m1_pred = self.model1.predict(user_ids, item_ids)
        m2_pred = self.model2.predict(user_ids, item_ids)
        return self.blend * m1_pred + (1 - self.blend) * m2_pred

class FunkSVD(Recommender):
    """
    Matrix Factorization using Surprise library.
    """
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=None, verbose=False):
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=random_state, verbose=verbose)

    def fit(self, train_df):
        reader = Reader()
        data = Dataset.load_from_df(train_df[['user_id','movie_id','rating']], reader)
        trainset = data.build_full_trainset()
        self.algo.fit(trainset)
        return self

    def predict(self, user_ids, item_ids):
        preds = []
        for u, i in zip(user_ids, item_ids):
            pred = self.algo.predict(u, i)
            preds.append(pred.est)
        return np.array(preds)

