import streamlit as st
from engine import KMeans, generate_data, silhouette_score, davies_bouldin_score
from plot import plot_kmeans_interactive, visualize_centroid_movements, plot_data
import pandas as pd

st.set_page_config(
    page_title="K-Means Clustering", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def display_algorithm_overview():
    with st.sidebar.expander("K-Means Algorithm Overview", expanded=False):
        st.markdown("""    
        1. **Initialize** K centroids:
        """)
        st.latex(r"\{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_K\}")
        st.markdown("2. **Assign each point** to the nearest centroid:")
        st.latex(r"\mathbf{x} \in \text{cluster } j \iff d(\mathbf{x}, \mathbf{c}_j) = \min_k d(\mathbf{x}, \mathbf{c}_k)")
        st.markdown("3. **Recompute the centroids:**")
        st.latex(r"\mathbf{c}_j = \frac{1}{|\mathcal{C}_j|} \sum_{\mathbf{x} \in \mathcal{C}_j} \mathbf{x}")
        st.markdown("4. **Repeat** until convergence.")

def sidebar_parameters():

    st.sidebar.subheader("Data Parameters")
    distribution = st.sidebar.selectbox(
        "Select Data Distribution", 
        ['Gaussian', 'Ring', 'Spiral', 'Moon', 'Uniform', 'Swiss Roll', 'Custom']
    )
    n_samples = st.sidebar.slider("Number of Samples", 10, 500, 100)
    n_dimensions = st.sidebar.selectbox("Number of Dimensions", [2, 3])

    st.sidebar.markdown("---")


    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
    init_strategy = st.sidebar.selectbox("Initialization Strategy", ['random', 'kmeans++'])
    method = st.sidebar.selectbox("Clustering Method", ['kmeans', 'kmedians', 'kmedoids'])
    metric = st.sidebar.selectbox("Distance Metric", ['euclidean', 'cosine', 'manhattan'])


    return distribution, n_clusters, n_samples, n_dimensions, metric, init_strategy, method


def generate_and_plot_data(distribution, n_samples, n_dimensions):
    data = generate_data(distribution, n_samples, n_dimensions)

    fig = plot_data(data, n_dimensions)

    return data, fig

def perform_clustering(data, n_clusters, metric, init_strategy, n_dimensions, method):
    
    kmeans = KMeans(n_clusters=n_clusters, metric=metric, init_strategy=init_strategy, method=method)
    labels = kmeans.fit(data)
    centroids = kmeans.centroids

    fig = plot_kmeans_interactive(data, labels, centroids, n_dimensions, metric=metric)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    inertia = kmeans.inertia_

    return kmeans, labels, fig, silhouette, davies_bouldin, inertia

def display_iteration_process(data, kmeans, labels, n_dimensions, metric):
    fig = visualize_centroid_movements(data, kmeans.iter_centroids, labels, n_dimensions, metric)
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("K-Means Clustering")

    display_algorithm_overview()

    distribution, n_clusters, n_samples, n_dimensions, metric, init_strategy, method = sidebar_parameters()

    st.session_state.setdefault('X', None)
    st.session_state.setdefault('fig_points', None)
    st.session_state.setdefault('fig_clustering', None)
    st.session_state.setdefault('metrics', None)


    if st.button("Generate Dataset"):
        st.session_state.X, st.session_state.fig_points = generate_and_plot_data(distribution, n_samples, n_dimensions)

    if st.session_state.fig_points:
        st.subheader("Generated Data Points")
        st.plotly_chart(st.session_state.fig_points, use_container_width=True)


    if st.button("Perform Clustering"):
        if st.session_state.X is None:
           st.error("Please generate a dataset first.")
        else:
            kmeans, labels, fig, silhouette, davies_bouldin, inertia = perform_clustering(
               st.session_state.X, n_clusters, metric, init_strategy, n_dimensions, method
            )
            st.session_state.update({"kmeans": kmeans, "labels": labels, "fig_clustering": fig})

            metrics_data = {
                    "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Inertia"],
                    "Description": [
                        "Measures how similar an object is to its own cluster compared to other clusters (higher is better).",
                        "Measures the average ratio of distances between clusters (lower is better).",
                        "Measures the sum of squared distances from each point to its assigned centroid (lower is better)."
                    ],
                    "Value": [silhouette, davies_bouldin, inertia],
            }
            metrics_table = pd.DataFrame(metrics_data)
            st.session_state.metrics = metrics_table

    if st.session_state.metrics is not None:   
        st.subheader("Clustering Metrics")
        st.table(st.session_state.metrics)

    if st.session_state.fig_clustering:
        st.subheader("Clustering Results")
        st.plotly_chart(st.session_state.fig_clustering, use_container_width=True)

        if st.checkbox("Show Details"):
            display_iteration_process(
                st.session_state.X, st.session_state.kmeans, 
                st.session_state.labels, n_dimensions, metric
            )

if __name__ == "__main__":
    main()
