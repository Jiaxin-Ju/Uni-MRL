"""NT-Xent loss shared by MolCLR pre-training and Uni-MRL alignment."""

import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, device=None, temperature=0.5, use_cosine_similarity=True):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.device = device
        self.temperature = float(temperature)
        self.use_cosine_similarity = use_cosine_similarity

    def _similarity(self, representations):
        if self.use_cosine_similarity:
            representations = F.normalize(representations, dim=1)
        return representations @ representations.T

    @staticmethod
    def _negative_mask(batch_size, device):
        size = 2 * batch_size
        mask = torch.ones((size, size), dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)
        indices = torch.arange(batch_size, device=device)
        mask[indices, indices + batch_size] = False
        mask[indices + batch_size, indices] = False
        return mask

    def forward(self, zis, zjs):
        if zis.ndim != 2 or zjs.ndim != 2 or zis.shape != zjs.shape:
            raise ValueError(
                f"NT-Xent expects equal [batch, feature] tensors, got {zis.shape} and {zjs.shape}"
            )
        batch_size = zis.size(0)
        if batch_size < 2:
            # A final singleton batch contains no negatives.
            return (zis.sum() + zjs.sum()) * 0.0

        representations = torch.cat((zjs, zis), dim=0)
        similarity = self._similarity(representations)
        positives = torch.cat(
            (
                torch.diag(similarity, batch_size),
                torch.diag(similarity, -batch_size),
            )
        ).reshape(2 * batch_size, 1)
        negatives = similarity[self._negative_mask(batch_size, similarity.device)].reshape(
            2 * batch_size, -1
        )
        logits = torch.cat((positives, negatives), dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
