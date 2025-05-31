import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from core.base_models import Recommender
from torch.optim.lr_scheduler import CosineAnnealingLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LightGCNRecommender(Recommender):
    def __init__(self, n_factors=64, n_layers=3, lr=0.1, batch_size=8192,
                 n_epochs=100, weight_decay=6e-5, alpha=0.2, default_rating=3.53):

        self.n_factors = n_factors
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.device = device
        self.alpha = alpha
        self.default_rating = default_rating
        self.model = None


    def fit(self, train_df):
        unique_users = train_df['user_id'].unique()
        unique_items = train_df['movie_id'].unique()

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        train_data = self._preprocess_data(train_df)
        self.R = self._create_interaction_matrix(train_data)
        train_loader = self._create_data_loader(train_data)

        self.model = LightGCN(self.n_users, self.n_items, self.n_factors,
                              self.n_layers, self.R, self.alpha).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=0.003)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(self.n_epochs), desc="Training LightGCN"):
            self.model.train()
            total_loss = 0

            for users, items, ratings in train_loader:
                users = users.to(self.device, non_blocking=True)
                items = items.to(self.device, non_blocking=True)
                ratings = ratings.float().to(self.device, non_blocking=True)

                optimizer.zero_grad()

                pred_ratings = self.model(users, items)
                loss = criterion(pred_ratings, ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

    def _preprocess_data(self, train_df):
        mapped_data = train_df.copy()
        mapped_data['user_idx'] = mapped_data['user_id'].map(self.user_mapping)
        mapped_data['item_idx'] = mapped_data['movie_id'].map(self.item_mapping)
        return mapped_data

    def _create_interaction_matrix(self, data):
        rows = np.concatenate([data['user_idx'].values, data['item_idx'].values + self.n_users])
        cols = np.concatenate([data['item_idx'].values + self.n_users, data['user_idx'].values])
        values = np.ones(len(rows))

        adj_mat = sp.coo_matrix((values, (rows, cols)),
                                shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        # Normalize the adjacency matrix
        degrees = np.array(adj_mat.sum(axis=1)).squeeze()
        d_inv_sqrt = np.power(degrees, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        norm_adj_coo = norm_adj.tocoo()

        indices = torch.LongTensor(np.vstack((norm_adj_coo.row, norm_adj_coo.col)))
        values = torch.FloatTensor(norm_adj_coo.data)
        shape = torch.Size(norm_adj_coo.shape)

        return torch.sparse_coo_tensor(indices, values, size=shape, device=self.device)
    def _create_data_loader(self, data):
        class InteractionDataset(Dataset):
            def __init__(self, data):
                self.users = torch.LongTensor(data['user_idx'].values)
                self.items = torch.LongTensor(data['item_idx'].values)
                self.ratings = torch.FloatTensor(data['rating'].values)

            def __len__(self):
                return len(self.users)

            def __getitem__(self, idx):
                return self.users[idx], self.items[idx], self.ratings[idx]

        dataset = InteractionDataset(data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def predict(self, user_ids, item_ids, batch_size=10000):
        self.model.eval()
        preds = []

        valid_mask = [
            (u in self.user_mapping and i in self.item_mapping)
            for u, i in zip(user_ids, item_ids)
        ]

        with torch.no_grad():
            for start_idx in tqdm(range(0, len(user_ids), batch_size), desc="Predicting w LightGCN"):
                end_idx = min(start_idx + batch_size, len(user_ids))
                batch_users = user_ids[start_idx:end_idx]
                batch_items = item_ids[start_idx:end_idx]
                batch_mask = valid_mask[start_idx:end_idx]

                user_indices = [self.user_mapping.get(u, 0) for u, m in zip(batch_users, batch_mask) if m]
                item_indices = [self.item_mapping.get(i, 0) for i, m in zip(batch_items, batch_mask) if m]

                if user_indices:
                    user_tensor = torch.LongTensor(user_indices).to(self.device)
                    item_tensor = torch.LongTensor(item_indices).to(self.device)
                    batch_preds = self.model(user_tensor, item_tensor).cpu().numpy()

                    pred_idx = 0
                    for is_valid in batch_mask:
                        if is_valid:
                            preds.append(batch_preds[pred_idx])
                            pred_idx += 1
                        else:
                            preds.append(self.default_rating)
                else:
                    preds.extend([self.default_rating] * len(batch_mask))

            return np.array(preds)

class LightGCN(nn.Module):
    """
    LightGCN model implementation with PyTorch
    """

    def __init__(self, n_users, n_items, n_factors, n_layers, norm_adj, alpha=0.2, dropout=0):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.alpha = alpha
        self.dropout = dropout

        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.53))

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        all_embeddings = self._get_ego_embeddings()
        all_emb_list = [all_embeddings]

        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            if self.alpha > 0:
                all_embeddings = (1 - self.alpha) * all_embeddings + self.alpha * all_emb_list[0]
            all_emb_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(all_emb_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])

        user_embeddings = user_all_embeddings[users]
        item_embeddings = item_all_embeddings[items]

        user_biases = self.user_bias(users).squeeze()
        item_biases = self.item_bias(items).squeeze()

        interaction = torch.sum(user_embeddings * item_embeddings, dim=1)

        ratings =  self.global_bias + user_biases + item_biases + interaction
        # ratings = 1 + 4 * torch.sigmoid(ratings / 3)
        # # ratings = 3.0 + 2.5 * torch.tanh(ratings / 2.5)
        return ratings

    def _get_ego_embeddings(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        return ego_embeddings


