"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch.nn as nn  # for pytorch based ops


class MF(nn.Module):
    """Model for Matrix Factorization"""
    def __init__(self, users: int, items: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(users, latent_dim)
        self.item_embed = nn.Embedding(items, latent_dim)

    def forward(self, user, items):
        user_emb, item_emb = self.user_embed(user), self.item_embed(items)

        return (user_emb * item_emb).sum(1)

    def save_embeddings(self, user, items):
        return self.user_embed(user), self.item_embed(items)
