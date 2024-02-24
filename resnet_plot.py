import os

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update
from keras.applications.resnet50 import ResNet50, preprocess_input
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("assets", exist_ok=True)


# Step 1: Preprocessing
def load_images_from_folder(folder):
    images = []
    file_paths = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("L")
        if img is not None:
            img_array = np.array(img).reshape(224, 224, 1)
            img_array = np.repeat(img_array, 3, axis=2)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            if not os.path.exists(f"assets/{filename}"):
                plt.imsave(f"assets/{filename}", img)

            asset_image_path = f"/assets/{filename}"
            file_paths.append(asset_image_path)

    return np.vstack(images), file_paths


# Set your path to the folder with images
folder_path = "phase_for_resnet_resized"
images, image_paths = load_images_from_folder(folder_path)

# Step 2: Feature Extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(images)

# Step 3: Dimensionality Reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(features)

# Step 4: Clustering
kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(pca_result)

# Create a DataFrame for Plotly
df = pd.DataFrame({
    'Component 1': pca_result[:, 0],
    'Component 2': pca_result[:, 1],
    'Cluster': kmeans.labels_,
    'Image': image_paths
})

df1 = pd.DataFrame({
    'Component 1': tsne_result[:, 0],
    'Component 2': tsne_result[:, 1],
    'Cluster': kmeans.labels_,
    'Image': image_paths
})

fig1 = px.scatter(df, x="Component 1", y="Component 2")
fig1.update_layout(height=1000, width=1000, title="PCA of ResNet50 Features")

fig = px.scatter(df1, x="Component 1", y="Component 2")
fig.update_layout(height=1000, width=1000, title="t-SNE of ResNet50 Features")

# Plotly App
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig1, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['Image']

    children = [
        html.Div(children=[
            html.Img(src=img_src, style={"width": "100%"}),
        ],
            style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=False)
