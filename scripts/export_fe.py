# scripts/export_pca.py
import torch
from joblib import load
import numpy as np

mean_vectors, pca_models = load("pca_models.joblib")
H = W = 14
C = next(iter(mean_vectors.values())).shape[0]
Kmax = max(p.components_.shape[0] for p in pca_models.values())

mean_t = torch.stack([
    torch.from_numpy(mean_vectors[(i, j)]).float()
    for i in range(H) for j in range(W)
], dim=0).view(H, W, C)

basis_t = torch.zeros((H, W, Kmax, C), dtype=torch.float32)
for i in range(H):
    for j in range(W):
        comps = pca_models[(i, j)].components_.astype(np.float32)
        basis_t[i, j, :comps.shape[0]] = torch.from_numpy(comps)

torch.save({"mean_t": mean_t, "basis_t": basis_t}, "src/principal_component_analysis/models/pca_tensors.pt")
print("Saved pca_tensors.pt")
