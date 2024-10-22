import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import calculate_metric_distance

def plot_kmeans_interactive(X, labels, centroids, n_dimensions, iter_centroids=None, metric=None, title=None):
    if n_dimensions == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:, 0], y=[0] * len(X), mode='markers', 
                                 marker=dict(color=labels, colorscale='viridis', size=10),
                                 name="Data Points"))
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=[0] * len(centroids), mode='markers',
                                 marker=dict(color='red', size=15, symbol='x'),
                                 name="Centroids"))
        fig.update_layout(title='K-Means Clustering (1D)', xaxis_title='Feature 1', showlegend=True)
        return fig

    if n_dimensions == 2:
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels, color_continuous_scale='viridis',
                         labels={'x': 'Feature 1', 'y': 'Feature 2'}, title='K-Means Clustering (2D)')
        fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                                 marker=dict(color='red', size=15, symbol='x'),
                                 name="Centroids"))

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = np.array([
            np.argmin([calculate_metric_distance(point, centroid, metric) for centroid in centroids]) 
            for point in grid_points
        ])
        Z = Z.reshape(xx.shape)

        fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=Z,
                                 colorscale='viridis', opacity=0.3, showscale=False))

        return fig


    elif n_dimensions == 3:
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels, color_continuous_scale='viridis',
                            labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'},
                            title='K-Means Clustering (3D)')
        fig.add_trace(go.Scatter3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                                   mode='markers', marker=dict(color='red', size=10, symbol='x'),
                                   name="Centroids"))
        return fig


def plot_data(data, n_dimensions):
    fig = None
    
    if n_dimensions == 1:
        fig = go.Figure(go.Scatter(
            x=data[:, 0], y=[0] * len(data), mode='markers', marker=dict(size=10)
        ))
        fig.update_layout(title='Generated Data Points (1D)', xaxis_title='Feature 1')
    elif n_dimensions == 2:
        fig = px.scatter(
            x=data[:, 0], y=data[:, 1], labels={'x': 'Feature 1', 'y': 'Feature 2'}, 
            title='Generated Data Points (2D)'
        )
    else:  # 3D case
        fig = px.scatter_3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'},
            title='Generated Data Points (3D)'
        )

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    return fig



def visualize_centroid_movements(X, iter_centroids, labels, n_dimensions, metric='euclidean'):
    fig = go.Figure()

    if n_dimensions == 2:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=labels, colorscale="viridis", size=8), name='Data Points'))
    elif n_dimensions == 3:
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(color=labels, colorscale="viridis", size=8), name='Data Points'))

    colors = px.colors.sequential.Viridis

    for i in range(len(iter_centroids) - 1):
        old_centroids = iter_centroids[i]
        new_centroids = iter_centroids[i + 1]

        for j in range(len(old_centroids)):
            cluster_label = int(labels[j])  
            color = colors[(cluster_label) % len(colors)]

            if n_dimensions == 2:
                
                fig.add_annotation(
                    x=new_centroids[j, 0], 
                    y=new_centroids[j, 1],
                    ax=old_centroids[j, 0], 
                    ay=old_centroids[j, 1],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor=color,
                    opacity=0.7,
                    arrowwidth=2
                )

            elif n_dimensions == 3:
                
                fig.add_trace(go.Scatter3d(
                    x=[old_centroids[j, 0], new_centroids[j, 0]], 
                    y=[old_centroids[j, 1], new_centroids[j, 1]], 
                    z=[old_centroids[j, 2], new_centroids[j, 2]],
                    mode='lines',
                    line=dict(color=color, width=2), 
                    name='Centroid Movement'
                ))

    for j in range(len(iter_centroids[0])):
        color = colors[int(labels[j]) % len(colors)] 
        if n_dimensions == 2:
            fig.add_trace(go.Scatter(
                x=[iter_centroids[0][j, 0]], 
                y=[iter_centroids[0][j, 1]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle-open', line=dict(color='black', width=2)),
                name=f'Start Centroid {j}'
            ))
        elif n_dimensions == 3:
            fig.add_trace(go.Scatter3d(
                x=[iter_centroids[0][j, 0]], 
                y=[iter_centroids[0][j, 1]], 
                z=[iter_centroids[0][j, 2]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle-open', line=dict(color='black', width=2)),
                name=f'Start Centroid {j}'
            ))

    for j in range(len(iter_centroids[-1])):
        color = colors[int(labels[j]) % len(colors)] 
        if n_dimensions == 2:
            fig.add_trace(go.Scatter(
                x=[iter_centroids[-1][j, 0]], 
                y=[iter_centroids[-1][j, 1]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(color='black', width=2)),
                name=f'End Centroid {j}'
            ))
        elif n_dimensions == 3:
            fig.add_trace(go.Scatter3d(
                x=[iter_centroids[-1][j, 0]], 
                y=[iter_centroids[-1][j, 1]], 
                z=[iter_centroids[-1][j, 2]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(color='black', width=2)),
                name=f'End Centroid {j}'
            ))

    fig.update_layout(title='Centroid Movements', showlegend=True)

    return fig