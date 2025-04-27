import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import scipy.sparse as sp
from imdb import IMDb
import pandas as pd
from tqdm import tqdm
from core.loader import load_movielens
from abc import ABC, abstractmethod

class Recommender(ABC):
    def fit(self, train_df):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for given user–item pairs.
        Returns a list/array of predictions: f(u, i) = r̂_ui
        """
        pass

    def evaluate(self, test_df):
        """Evaluate model using specified metrics."""
        user_ids = test_df["user_id"].values
        item_ids = test_df["movie_id"].values
        true_ratings = test_df["rating"].values
        preds = self.predict(user_ids, item_ids)

        rmse = root_mean_squared_error(true_ratings, preds)
        r2 = r2_score(true_ratings, preds)

        return preds, true_ratings,  {"rmse": rmse, "r2": r2}


class UserCFRecommender(Recommender):
    """
    User based Collaborative Filtering
    """

    def __init__(self, default_rating=3.0, eps=1e-12, k_neighbors=None):
        """
        :param default_rating: fallback rating when no info is available
        :param eps: small constant to avoid division by zero
        """
        self.default_rating = default_rating
        self.eps = eps
        self.k_neighbors = k_neighbors

    def fit(self, train_df):
        """
        Build rating matrix and user similarity.
        :param train_df: DataFrame with columns ['user_id','movie_id','rating']
        """
        self.R = train_df.pivot(index='user_id', columns='movie_id', values='rating')

        self.user_means = self.R.mean(axis=1)
        R_centered = self.R.sub(self.user_means, axis=0).fillna(0)
        sim = cosine_similarity(R_centered)
        self.sim_df = pd.DataFrame(sim, index=self.R.index, columns=self.R.index)
        self.R = self.R.fillna(0)

    def predict(self, user_ids, item_ids):
        preds = []
        k = self.k_neighbors

        for u, i in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Predicting w User-CF"):
            if u not in self.R.index or i not in self.R.columns:
                preds.append(self.default_rating)
                continue

            ratings_i = self.R[i]
            neighbors = ratings_i[ratings_i > 0].index  # users who rated item i

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
    Item based Collaborative Filtering
    """
    def __init__(self, default_rating=3.0, eps=1e-12, k_neighbors=None):
        self.default_rating = default_rating
        self.eps = eps
        self.k_neighbors = k_neighbors

    def fit(self, train_df: pd.DataFrame):
        """
        Builds item-item similarity matrix using raw ratings.
        :param train_df: DataFrame with columns ['user_id','movie_id','rating']
        """
        self.R = train_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

        sim = cosine_similarity(self.R.T)
        self.sim_df = pd.DataFrame(sim, index=self.R.columns, columns=self.R.columns)

        self.R = self.R.fillna(0)

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

    def __init__(self, default_rating=3.0, eps=1e-8, k_neighbors=None):
            self.default_rating = default_rating
            self.eps = eps
            self.k_neighbors = k_neighbors

    def fit(self, train_df, items_df, feature_cols):
        """
        Precompute item-item similarity and store user ratings.

        :param train_df: DataFrame with ['user_id','movie_id','rating']
        :param items_df: DataFrame with ['movie_id'] + feature_cols
        :param feature_cols: list of metadata column names to use
        """
        # 1) user-item rating matrix
        self.user_ratings = train_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        # 2) per-user mean (ignoring zeros)
        self.user_means = (
            self.user_ratings
            .replace(0, np.nan)
            .mean(axis=1)
            .fillna(self.default_rating)
        )

        # 3) build item-profile matrix
        self.item_profiles = (
            items_df
            .set_index("movie_id")[feature_cols]
            .fillna(0)
        )

        # 4) compute cosine similarity between items
        sim = cosine_similarity(self.item_profiles.values)
        self.item_sim = pd.DataFrame(
            sim,
            index=self.item_profiles.index,
            columns=self.item_profiles.index
        )

    def predict(self, user_ids, item_ids):
        """
        predict ratings for given user-item pairs.

        :param user_ids: array-like of user_id
        :param item_ids: array-like of movie_id
        :return: numpy array of predicted ratings
        """
        preds = []
        for u, i in tqdm(zip(user_ids, item_ids), total=len(user_ids), desc="Predicting w CBF"):
            # fallback for unknown user or item
            if u not in self.user_ratings.index or i not in self.item_sim.index:
                preds.append(self.default_rating)
                continue

            ratings_u = self.user_ratings.loc[u]
            rated = ratings_u[ratings_u > 0]
            if rated.empty:
                # user has no history
                preds.append(self.user_means.get(u, self.default_rating))
                continue

            sims = self.item_sim.loc[i, rated.index]
            deviations = rated - self.user_means[u]

            if self.k_neighbors is not None and len(sims) > self.k_neighbors:
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
    Hybrid of two recommenders:
    \u03BB * CF + (1 - \u03BB) * CBF
    """

    def __init__(self, cf_model, cbf_model, blend=0.5):
        """
        :param cf_model: collaborative filtering model (must have .predict method)
        :param cbf_model: content-based model (must have .predict method)
        :param blend: weight for CF (0.0 = only CBF, 1.0 = only CF)
        """
        self.cf = cf_model
        self.cbf = cbf_model
        self.blend = blend

    def predict(self, user_ids, item_ids):
        cf_pred = self.cf.predict(user_ids, item_ids)
        cbf_pred = self.cbf.predict(user_ids, item_ids)
        return self.blend * cf_pred + (1 - self.blend) * cbf_pred


