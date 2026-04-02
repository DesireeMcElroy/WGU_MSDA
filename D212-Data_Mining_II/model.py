wider_range_cols = [
    'population',
    'income',
    'total_charges',
    'additional_charges'
]

# scale only wider range columns
scaler = RobustScaler()
print(f'Using Robust Scaler on only {(wider_range_cols)}')

scaled_wide = pd.DataFrame(
    scaler.fit_transform(df[wider_range_cols]),
    columns=wider_range_cols,
    index=df.index
)

# merge scaled columns back, replacing originals
scaled_df = df.drop(columns=wider_range_cols).join(scaled_wide)

# scaled_df.head()


# elbow + silhouette scan
inertia, sil_scores = [], []
for k in range(2, 12):
    km = KMeans(n_clusters=k, random_state=42).fit(scaled_df)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_df, km.labels_))
    print(f'k={k}  inertia={km.inertia_:,.0f}  silhouette={round(silhouette_score(scaled_df, km.labels_), 4)}')

plt.title('Elbow Method')
plt.plot(range(2, 12), inertia, marker='o')
plt.xlabel("Clusters")
plt.ylabel("WCSS (Inertia)")
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.show()

# --- Final model: k=4 ---
N_CLUSTERS = 4
km_final = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(scaled_df)
print(f'\nFinal model  k={N_CLUSTERS}  silhouette={round(silhouette_score(scaled_df, km_final.labels_), 4)}')

# cluster profiling dataframe (unscaled values + cluster label)
cluster_df = df.copy()
cluster_df['cluster'] = km_final.labels_
profile = cluster_df.groupby('cluster').agg(['mean', 'median', 'std'])
print(profile)

# boxplots: one subplot per feature, all 4 clusters side-by-side
features = [c for c in cluster_df.columns if c != 'cluster']
n_cols = 3
n_rows = -(-len(features) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
axes = axes.flatten()

for ax, feat in zip(axes, features):
    cluster_df.boxplot(column=feat, by='cluster', ax=ax, grid=False)
    ax.set_title(feat)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('')

# hide any unused subplots
for ax in axes[len(features):]:
    ax.set_visible(False)

fig.suptitle('Feature Distributions by Cluster', fontsize=14)
plt.tight_layout()
plt.show()