import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import umap
import pandas as pd
import streamlit as st

def correlation_heatmap(df, x_cols):
    fig, ax = plt.subplots()
    corr = df[x_cols].corr()
    sns.heatmap(corr, annot=True, cmap='viridis', ax=ax)
    st.pyplot(fig)

def pca_analysis(df, x_cols, color_labels=None):
    st.subheader("PCA (Principal Component Analysis)")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df[x_cols])
    pc_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    if color_labels is not None:
        pc_df['label'] = color_labels
        fig = px.scatter(pc_df, x='PC1', y='PC2', color='label', title="PCA Scatter Plot")
    else:
        fig = px.scatter(pc_df, x='PC1', y='PC2', title="PCA Scatter Plot")
    st.plotly_chart(fig)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

def kmeans_clustering(df, cluster_cols, n_clusters=2, reduction_method='PCA'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(df[cluster_cols])
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels

    st.subheader("Pairwise Scatter Matrix (Colored by Cluster)")
    fig_matrix = px.scatter_matrix(df_clustered, dimensions=cluster_cols, color='Cluster', title='KMeans Clustering Scatter Matrix')
    st.plotly_chart(fig_matrix)
    st.write("Cluster Counts:", pd.Series(cluster_labels).value_counts())

    st.subheader(f"2D Cluster Visualization ({reduction_method})")
    if reduction_method == 'PCA':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(df[cluster_cols])
        embed_df = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    elif reduction_method == 'UMAP':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(df[cluster_cols])
        embed_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    else:
        raise ValueError("Unsupported reduction method")
    embed_df['Cluster'] = cluster_labels
    embed_df['Index'] = df.index

    # Merge embed_df with df for complete hover_data columns
    merged = embed_df.merge(df.reset_index(), left_on='Index', right_on='index', suffixes=('', '_orig'))

    if reduction_method == 'PCA':
        fig_2d = px.scatter(
            merged, x='PC1', y='PC2', color='Cluster',
            hover_data=['Index'] + list(df.columns),
            title='KMeans Clusters (PCA 2D)'
        )
    else:
        fig_2d = px.scatter(
            merged, x='UMAP1', y='UMAP2', color='Cluster',
            hover_data=['Index'] + list(df.columns),
            title='KMeans Clusters (UMAP 2D)'
        )
    st.plotly_chart(fig_2d)

    return cluster_labels, embed_df

def knn_neighbors(df, cluster_cols, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(df[cluster_cols])
    distances, indices = nbrs.kneighbors(df[cluster_cols])
    st.write(f"Mean distance to {n_neighbors} nearest neighbors (first 10):", distances[:,1:].mean(axis=1)[:10])

def umap_embedding(df, cluster_cols, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(df[cluster_cols])
    embed_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    fig = px.scatter(embed_df, x='UMAP1', y='UMAP2', title="UMAP Embedding")
    st.plotly_chart(fig)
    return embed_df

def pca_umap_embedding(df, cluster_cols, n_pca=20, n_neighbors=15, min_dist=0.1):
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(df[cluster_cols])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_pca)
    embed_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    return embed_df, pca.explained_variance_ratio_

# 2D generic cluster plot (if needed for later)
def cluster_2d_plot(embed_df, cluster_labels, df, cluster_colname='Cluster'):
    embed_df[cluster_colname] = cluster_labels
    embed_df['Index'] = df.index
    fig = px.scatter(
        embed_df, 
        x=embed_df.columns[0], 
        y=embed_df.columns[1], 
        color=cluster_colname,
        hover_data=['Index'] + df.columns.tolist(),
        title=f'2D Cluster Visualization ({embed_df.columns[0]} vs {embed_df.columns[1]})'
    )
    return fig, embed_df
from streamlit_plotly_events import plotly_events

def interactive_knn_umap_plot(embed_df, df, label_col="Manual_Cluster_Label"):
    # Add a unique index for selection (for non-default index DataFrames)
    embed_df = embed_df.copy()
    embed_df['Index'] = df.index

    fig = px.scatter(
        embed_df,
        x=embed_df.columns[0], y=embed_df.columns[1],
        hover_data=['Index'] + df.columns.tolist(),
        title="Interactive Embedding (Click a point to inspect/label)"
    )
    # Use streamlit-plotly-events to capture click
    selected_points = plotly_events(fig, click_event=True, hover_event=False)
    selected_row = None

    if selected_points:
        idx = selected_points[0]['pointIndex']
        row_index = embed_df.iloc[idx]['Index']
        selected_row = df.loc[row_index]
        st.markdown(f"**Selected data point (row index: {row_index}):**")
        st.write(selected_row)
        # Label input
        new_label = st.text_input("Enter/assign a cluster label for this point (or leave blank):", key=f"label_{row_index}")
        if new_label:
            st.session_state.setdefault(label_col, {})
            st.session_state[label_col][row_index] = new_label
            st.success(f"Labeled row {row_index} as '{new_label}'.")
    else:
        st.info("Click on a point in the plot above to inspect and label it.")

    return st.session_state.get(label_col, {})
