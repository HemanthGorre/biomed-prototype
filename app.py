import streamlit as st
from modules import data_loader, preprocessing, eda, modeling
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Biomedical ML Prototype", layout="wide")
st.title("🧬 Biomedical Data Science Platform")

df = data_loader.upload_data()
if df is not None:
    data_loader.data_overview(df)
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    all_cols = df.columns.tolist()

    # ---- Step 1: Ask the user if the data is already labeled ----
    st.sidebar.header("Project Workflow Setup")
    is_labeled = st.sidebar.radio(
        "Is your uploaded data already labeled for supervised learning?",
        ["No, perform EDA/clustering first", "Yes, proceed to supervised modeling"],
        index=0
    )

    if is_labeled.startswith("Yes"):
        # ---- Two Tabs: EDA & Supervised ----
        tab1, tab2 = st.tabs(["Exploratory Data Analysis (EDA)", "Supervised Modeling"])
        with tab1:
            st.header("Unsupervised EDA: Clustering & Dimensionality Reduction")
            cluster_cols = st.multiselect("Choose columns for clustering/UMAP", options=numeric_cols, key='eda_tab_cluster_cols')
            if cluster_cols:
                method = st.radio("Select Unsupervised Method",
                                  ["KMeans Clustering", "KNN (Nearest Neighbors)", "PCA Only", "UMAP Only", "PCA → UMAP"], key='eda_tab_unsup_method')
                if method == "KMeans Clustering":
                    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=2, key='eda_tab_kmeans_nclust')
                    reduction_method = st.selectbox("2D visualization method", ["PCA", "UMAP", "PCA → UMAP"], key='eda_tab_kmeans_red')
                    cluster_labels, embed_df = eda.kmeans_clustering(df, cluster_cols, n_clusters=n_clusters, reduction_method=reduction_method)
                    embed_df['Index'] = df.index
                    unique_clusters = sorted(embed_df['Cluster'].unique())
                    selected_cluster = st.selectbox("Highlight and inspect which cluster?", unique_clusters, key="eda_tab_pca_cluster")
                    highlight_df = embed_df[embed_df['Cluster'] == selected_cluster]
                    st.write(f"**Data points in Cluster {selected_cluster}:**")
                    st.dataframe(df.loc[highlight_df['Index']])
                elif method == "KNN (Nearest Neighbors)":
                    n_neighbors = st.slider("Number of neighbors (K)", min_value=2, max_value=20, value=5, key='eda_tab_knn_k')
                    eda.knn_neighbors(df, cluster_cols, n_neighbors)
                    embed_df = eda.umap_embedding(df, cluster_cols, n_neighbors=15, min_dist=0.1)
                    manual_labels = eda.interactive_knn_umap_plot(embed_df, df)
                    if manual_labels:
                        df['Manual_Cluster_Label'] = df.index.map(lambda idx: manual_labels.get(idx, ""))
                        st.write("**Dataframe with manual cluster labels (KNN/UMAP EDA):**")
                        st.dataframe(df)
                        if 'Manual_Cluster_Label' in df.columns:
                            st.subheader("Manual Cluster Label Visualization")
                            embed_df['Manual_Cluster_Label'] = df['Manual_Cluster_Label']
                            fig = px.scatter(embed_df, x=embed_df.columns[0], y=embed_df.columns[1], color='Manual_Cluster_Label',
                                             title="Embedding Colored by Manual Cluster Label")
                            st.plotly_chart(fig)
                elif method == "UMAP Only":
                    n_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=50, value=15, key='eda_tab_umap_nn')
                    min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='eda_tab_umap_md')
                    embed_df = eda.umap_embedding(df, cluster_cols, n_neighbors, min_dist)
                    manual_labels = eda.interactive_knn_umap_plot(embed_df, df)
                    if manual_labels:
                        df['Manual_Cluster_Label'] = df.index.map(lambda idx: manual_labels.get(idx, ""))
                        st.write("**Dataframe with manual cluster labels (UMAP EDA):**")
                        st.dataframe(df)
                        if 'Manual_Cluster_Label' in df.columns:
                            st.subheader("Manual Cluster Label Visualization")
                            embed_df['Manual_Cluster_Label'] = df['Manual_Cluster_Label']
                            fig = px.scatter(embed_df, x=embed_df.columns[0], y=embed_df.columns[1], color='Manual_Cluster_Label',
                                             title="Embedding Colored by Manual Cluster Label")
                            st.plotly_chart(fig)
                elif method == "PCA Only":
                    eda.pca_analysis(df, cluster_cols)
                elif method == "PCA → UMAP":
                    n_pca = st.slider("PCA components before UMAP", 2, min(50, len(cluster_cols)), 20, key='eda_tab_pcaumap_npca')
                    n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15, key='eda_tab_pcaumap_nn')
                    min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.01, key='eda_tab_pcaumap_md')
                    embed_df, explained_var = eda.pca_umap_embedding(df, cluster_cols, n_pca, n_neighbors, min_dist)
                    st.subheader("PCA→UMAP Embedding (All Data)")
                    fig_umap = px.scatter(embed_df, x="UMAP1", y="UMAP2", title="PCA→UMAP Embedding")
                    st.plotly_chart(fig_umap)
                    st.write("PCA Explained Variance (first n components):", explained_var[:n_pca].sum())
        with tab2:
            # User selects Y
            y_col, x_cols = data_loader.select_variables(df)
            if y_col and x_cols:
                st.sidebar.success(f"Y: {y_col}, X: {x_cols}")
                st.write(f"Selected dependent variable (Y): `{y_col}`")
                st.write(f"Selected independent variables (X): `{x_cols}`")

                st.header("Step 3: Data Preprocessing & Normalization")
                scaler_choice = st.selectbox("Choose normalization method", ["StandardScaler", "Min-Max Scaler", "RobustScaler"])
                X_scaled = preprocessing.preprocess_data(df, x_cols, scaler_choice)
                st.write(f"Preview of data after {scaler_choice}:")
                st.dataframe(X_scaled.head())

                st.header("Step 4: Exploratory Analysis")
                st.subheader("Correlation Heatmap")
                eda.correlation_heatmap(df, x_cols)

                st.subheader("Principal Component Analysis (PCA)")
                eda.pca_analysis(df, x_cols)

                st.header("Step 5: Modeling & Evaluation")
                model_choice = st.selectbox("Choose ML Model",
                                            ["Linear Regression", "Random Forest", "XGBoost"] if not modeling.is_classification(df, y_col) else ["Logistic Regression", "Random Forest", "XGBoost"])
                metrics, details = modeling.train_and_evaluate(df.assign(**{col: X_scaled[col] for col in x_cols}), x_cols, y_col, model_choice, scaler_choice)
                st.write("**Model Evaluation Metrics:**")
                st.write(metrics)

                if details[-1] == 'classification' and model_choice != "Linear Regression" and details[2].nunique() == 2:
                    st.subheader("ROC Curve")
                    modeling.plot_roc_curve(*details[:3])

                # ----------- Explainability -----------
                with st.expander("🔍 Model Explainability: Feature Importance & SHAP"):
                    model, X_test, y_test, y_pred, task_type, used_features = details
                    # Feature Importance/Coef
                    if model_choice in ["Random Forest", "XGBoost"]:
                        st.subheader("Feature Importance (Model-based)")
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            importance_df = pd.DataFrame({"Feature": used_features, "Importance": importances}).sort_values("Importance", ascending=False)
                            st.bar_chart(importance_df.set_index("Feature"))
                            st.dataframe(importance_df)
                        else:
                            st.write("Feature importance not available for this model.")
                    elif model_choice in ["Linear Regression", "Logistic Regression"]:
                        st.subheader("Model Coefficients")
                        if hasattr(model, 'coef_'):
                            coefs = model.coef_.flatten() if hasattr(model, 'coef_') else []
                            coef_df = pd.DataFrame({"Feature": used_features, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)
                            st.bar_chart(coef_df.set_index("Feature"))
                            st.dataframe(coef_df)
                        else:
                            st.write("Coefficients not available for this model.")
                    # SHAP
                    st.subheader("SHAP Values (Model Explainability)")
                    try:
                        background = X_test.iloc[:200] if isinstance(X_test, pd.DataFrame) else X_test[:200]
                        explainer = None
                        if model_choice in ["Random Forest", "XGBoost"]:
                            explainer = shap.TreeExplainer(model)
                        elif model_choice in ["Linear Regression", "Logistic Regression"]:
                            explainer = shap.LinearExplainer(model, background)
                        if explainer is not None:
                            shap_values = explainer.shap_values(background)
                            st.write("SHAP summary plot (global feature importance):")
                            plt.figure()
                            shap.summary_plot(shap_values, background, feature_names=used_features, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()
                            st.write("Explore a single prediction (local explanation):")
                            sample_idx = st.number_input("Pick a row index for local SHAP explanation:", min_value=0, max_value=len(background)-1, value=0)
                            shap.initjs()
                            force_html = shap.plots.force(
                                explainer.expected_value,
                                shap_values[sample_idx],
                                background.iloc[sample_idx],
                                matplotlib=False, show=False
                            )
                            st.components.v1.html(force_html.html(), height=300)
                    except Exception as e:
                        st.warning(f"Error in SHAP explainability: {e}")

                st.header("Step 6: Comparative Evaluation (All Methods)")
                st.info("Evaluating all normalization and model combinations. This may take a moment.")
                results_df = modeling.compare_models_and_preprocessing(df, x_cols, y_col)
                st.write("### Comparison Table (All Normalizations & Models)")
                st.dataframe(results_df)
            else:
                st.info("Please select dependent and independent variables.")

    else:
        # --- One Tab (EDA, Unsupervised, and Supervised Reveal) ---
        st.header("Unsupervised EDA: Clustering & Dimensionality Reduction")
        cluster_cols = st.multiselect("Choose columns for clustering/UMAP", options=numeric_cols, key='eda_cluster_cols')
        if cluster_cols:
            method = st.radio("Select Unsupervised Method",
                              ["KMeans Clustering", "KNN (Nearest Neighbors)", "PCA Only", "UMAP Only", "PCA → UMAP"], key='unsup_method')

            supervised_unlocked = False

            if method == "KMeans Clustering":
                n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=2, key='kmeans_nclust')
                reduction_method = st.selectbox("2D visualization method", ["PCA", "UMAP", "PCA → UMAP"], key='kmeans_red')
                cluster_labels, embed_df = eda.kmeans_clustering(df, cluster_cols, n_clusters=n_clusters, reduction_method=reduction_method)
                embed_df['Index'] = df.index
                unique_clusters = sorted(embed_df['Cluster'].unique())
                selected_cluster = st.selectbox("Highlight and inspect which cluster?", unique_clusters, key="pca_cluster")
                highlight_df = embed_df[embed_df['Cluster'] == selected_cluster]
                st.write(f"**Data points in Cluster {selected_cluster}:**")
                st.dataframe(df.loc[highlight_df['Index']])
                label_for_supervised = st.radio(
                    "Do you want to use the detected clusters as labels for supervised learning?",
                    ["No", "Yes"], index=0, key='label_supervised'
                )
                if label_for_supervised == "Yes":
                    if 'Cluster_Label' not in df.columns:
                        df['Cluster_Label'] = pd.Series(cluster_labels, index=df.index)
                    st.success("Cluster labels assigned! You can now use these as the target (Y) for supervised learning below.")
                    supervised_unlocked = True

            # (repeat for other clustering methods as needed...)

            if ('Cluster_Label' in df.columns) and (supervised_unlocked):
                st.header("Supervised Learning on Labeled Clusters")
                x_cols_sl = st.multiselect(
                    "Select features (X) for supervised model",
                    [col for col in numeric_cols if col not in ['Cluster_Label']],
                    key='sl_x_cols'
                )
                if x_cols_sl:
                    y_col_sl = 'Cluster_Label'
                    st.write(f"Training supervised models to predict **cluster group** ({y_col_sl}) from features: {x_cols_sl}")

                    scaler_choice = st.selectbox("Choose normalization method for supervised", ["StandardScaler", "Min-Max Scaler", "RobustScaler"], key='sl_scaler')
                    X_scaled_sl = preprocessing.preprocess_data(df, x_cols_sl, scaler_choice)
                    st.write(f"Preview of data after {scaler_choice}:")
                    st.dataframe(X_scaled_sl.head())

                    st.header("Modeling & Evaluation (Cluster Label as Y)")
                    model_choice_sl = st.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest", "XGBoost"], key='sl_model')
                    metrics_sl, details_sl = modeling.train_and_evaluate(
                        df.assign(**{col: X_scaled_sl[col] for col in x_cols_sl}),
                        x_cols_sl, y_col_sl, model_choice_sl, scaler_choice
                    )
                    st.write("**Model Evaluation Metrics:**")
                    st.write(metrics_sl)
                    if details_sl[-1] == 'classification' and model_choice_sl != "Linear Regression" and details_sl[2].nunique() == 2:
                        st.subheader("ROC Curve")
                        modeling.plot_roc_curve(*details_sl[:3])

                    # ----------- Explainability -----------
                    with st.expander("🔍 Model Explainability: Feature Importance & SHAP"):
                        model, X_test, y_test, y_pred, task_type, used_features = details_sl
                        # Feature Importance/Coef
                        if model_choice_sl in ["Random Forest", "XGBoost"]:
                            st.subheader("Feature Importance (Model-based)")
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                importance_df = pd.DataFrame({"Feature": used_features, "Importance": importances}).sort_values("Importance", ascending=False)
                                st.bar_chart(importance_df.set_index("Feature"))
                                st.dataframe(importance_df)
                            else:
                                st.write("Feature importance not available for this model.")
                        elif model_choice_sl in ["Linear Regression", "Logistic Regression"]:
                            st.subheader("Model Coefficients")
                            if hasattr(model, 'coef_'):
                                coefs = model.coef_.flatten() if hasattr(model, 'coef_') else []
                                coef_df = pd.DataFrame({"Feature": used_features, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)
                                st.bar_chart(coef_df.set_index("Feature"))
                                st.dataframe(coef_df)
                            else:
                                st.write("Coefficients not available for this model.")
                        # SHAP
                        st.subheader("SHAP Values (Model Explainability)")
                        try:
                            background = X_test.iloc[:200] if isinstance(X_test, pd.DataFrame) else X_test[:200]
                            explainer = None
                            if model_choice_sl in ["Random Forest", "XGBoost"]:
                                explainer = shap.TreeExplainer(model)
                            elif model_choice_sl in ["Linear Regression", "Logistic Regression"]:
                                explainer = shap.LinearExplainer(model, background)
                            if explainer is not None:
                                shap_values = explainer.shap_values(background)
                                st.write("SHAP summary plot (global feature importance):")
                                plt.figure()
                                shap.summary_plot(shap_values, background, feature_names=used_features, show=False)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                st.write("Explore a single prediction (local explanation):")
                                sample_idx = st.number_input("Pick a row index for local SHAP explanation:", min_value=0, max_value=len(background)-1, value=0)
                                shap.initjs()
                                force_html = shap.plots.force(
                                    explainer.expected_value,
                                    shap_values[sample_idx],
                                    background.iloc[sample_idx],
                                    matplotlib=False, show=False
                                )
                                st.components.v1.html(force_html.html(), height=300)
                        except Exception as e:
                            st.warning(f"Error in SHAP explainability: {e}")

        else:
            st.info("Select at least two numeric columns for clustering or dimensionality reduction.")

else:
    st.warning("Please upload a CSV or Excel file to begin.")
