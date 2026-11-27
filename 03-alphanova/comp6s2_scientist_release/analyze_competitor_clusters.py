import zipfile, io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

with zipfile.ZipFile("D:/Projects/Quant/quant-competition-research/03-alphanova/city_coords_s2.zip") as z:
    with z.open("signal_cities_ambient_coords.parquet") as f:
        df = pd.read_parquet(io.BytesIO(f.read()))

print(f"Total competitor signals in database: {len(df)}")
coord_cols = [f"city_dim_{i}" for i in range(19)]
X = df[coord_cols].values

# Normalize to ensure unit vectors
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.maximum(norms, 1e-8)

print(f"City Dim 0 (IC proxy) mean: {X[:, 0].mean():.4f}, std: {X[:, 0].std():.4f}, max: {X[:, 0].max():.4f}, min: {X[:, 0].min():.4f}")

# Pairwise angular distance distribution across competitors
# Pick random sample of 100 to compute pairwise
dots = np.clip(X_norm @ X_norm.T, -1.0, 1.0)
angles = np.arccos(dots) * (180.0 / np.pi)
# Exclude diagonal
np.fill_diagonal(angles, np.nan)

print(f"Mean pairwise angular distance across competitors: {np.nanmean(angles):.1f}°")
print(f"Median pairwise angular distance: {np.nanmedian(angles):.1f}°")
print(f"Min pairwise distance (most crowded): {np.nanmin(angles):.2f}°")
print(f"Max pairwise distance (most opposed): {np.nanmax(angles):.1f}°")

# Cluster into major signal archetypes
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_norm)
df["cluster"] = kmeans.labels_

print("\n--- Competitor Signal Archetypes (6 Clusters) ---")
for c in range(6):
    cluster_df = df[df["cluster"] == c]
    centroid = kmeans.cluster_centers_[c]
    print(f"Cluster {c}: {len(cluster_df)} signals ({len(cluster_df)/len(df)*100:.1f}%) | Centroid Dim0: {centroid[0]:+.4f} | Top signals: {cluster_df['predictor_id'].iloc[:3].tolist()}")
