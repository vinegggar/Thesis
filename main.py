import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from gnn_model import GNNRecommender
from core.base_models import UserCFRecommender, ItemCFRecommender, CBFRecommender, HybridRecommender
from core.loader import load_movielens

# =======================
# LOAD DATA
# =======================

ratings, movies, users = load_movielens("datasets/ml-100k")
genre_cols = [c for c in movies.columns if movies[c].isin([0, 1]).all()]


# =======================
# SPLIT DATA
# =======================

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


train_df, test_df = split_data(ratings)

# =======================
# TRAIN AND EVALUATE MODELS
# =======================

user_cf_model = UserCFRecommender(k_neighbors=31)

item_cf_model = ItemCFRecommender(k_neighbors=20)

cbf_model = CBFRecommender(k_neighbors=20)

blend_value = 0.78
hybrid_model = HybridRecommender(user_cf_model, cbf_model, blend=blend_value)

# Now all models
models = {
    "UserCF (k=31)": user_cf_model,
    "ItemsCF (k=20)": item_cf_model,
    "CBF (k=20)": cbf_model,
    "Hybrid (CF+CBF)": hybrid_model
}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Fit
    if isinstance(model, CBFRecommender):
        model.fit(train_df, movies, feature_cols=genre_cols + ["year_norm"])
    else:
        model.fit(train_df)

    # Evaluate
    preds, true, metrics = model.evaluate(test_df)
    results[name] = metrics  # <-- Save metrics here

    plt.figure(figsize=(6, 6))
    plt.scatter(true, preds, alpha=0.3)
    plt.plot([0.5, 5.5], [0.5, 5.5], color="red", linestyle="--")
    plt.title(f"True vs Predicted ({name})")
    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    plt.grid()
    plt.axis("equal")
    plt.show()

# Print final table
print("\nEvaluation Results:")
for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")