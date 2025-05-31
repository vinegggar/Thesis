import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.base_models import UserCFRecommender, ItemCFRecommender, CBFRecommender, HybridRecommender, FunkSVD
from core.lightgcn import LightGCNRecommender
from core.loader import load_movielens

def split_data(df, test_ratio=0.2):
    train_list, test_list = [], []
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        test_ratings = group.sample(n=n_test, random_state=42)
        train_ratings = group.drop(test_ratings.index)
        train_list.append(train_ratings)
        test_list.append(test_ratings)
    return pd.concat(train_list), pd.concat(test_list)

def plot_predictions(true, preds, model_name, metrics):
    plt.figure(figsize=(8, 8))

    df = pd.DataFrame({'true': true, 'pred': preds})

    sampled_data = []
    samples_per_rating = 500

    for rating in [1, 2, 3, 4, 5]:
        rating_data = df[df['true'] == rating]
        if len(rating_data) > 0:
            n_samples = min(samples_per_rating, len(rating_data))
            sampled = rating_data.sample(n=n_samples, random_state=42)
            sampled_data.append(sampled)

    if sampled_data:
        plot_data = pd.concat(sampled_data)
        jittered_true = plot_data['true'] + np.random.normal(0, 0.05, len(plot_data))
        plt.scatter(jittered_true, plot_data['pred'], alpha=0.5, s=12)

    plt.plot([0, 6], [0, 6], 'r--', linewidth=2, label='Perfect')

    plt.xlim(0.5, 5.5)
    plt.ylim(0, 6)
    plt.xticks([1, 2, 3, 4, 5])

    plt.xlabel('True Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f"{model_name}\nRMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Load and split data
ratings, movies, users = load_movielens("datasets/ml-100k")
genre_cols = [c for c in movies.columns if movies[c].isin([0, 1]).all()]
train_df, test_df = split_data(ratings)

# Initialize models
user_cf_model = UserCFRecommender(k_neighbors=31)

item_cf_model = ItemCFRecommender(k_neighbors=20)

cbf_model = CBFRecommender(k_neighbors=20)

blend_value = 0.78
hybrid_model = HybridRecommender(user_cf_model, cbf_model, blend=blend_value)

funk_svd_model = FunkSVD(n_factors=100, n_epochs=40, lr_all=0.01, reg_all=0.1, verbose=False)

lightgcnmodel = LightGCNRecommender(n_factors=64, n_layers=3, n_epochs=100)

models = {
    "UserCF (k=31)": user_cf_model,
    "ItemsCF (k=20)": item_cf_model,
    "CBF (k=20)": cbf_model,
    "Hybrid (CF+CBF)": hybrid_model,
    "FunkSVD": funk_svd_model,
    'LightGCN': lightgcnmodel
}

# Train and evaluate
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    if isinstance(model, CBFRecommender):
        model.fit(train_df, movies, feature_cols=genre_cols + ["year_norm"])
    else:
        model.fit(train_df)

    preds, true, metrics = model.evaluate(test_df)
    results[name] = metrics

    plot_predictions(true, preds, name, metrics)