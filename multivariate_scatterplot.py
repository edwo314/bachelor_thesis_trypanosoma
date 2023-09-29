import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import ray
import scipy
import skimage
import umap.umap_ as umap
from dash import dcc, html, Input, Output, no_update, Dash
from joblib import load
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tryptag import TrypTag, CellLine
from tryptag.tryptools import cell_signal_analysis
from tryptag.tryptools.tryptools import cell_morphology_analysis, _mask_pruned_skeleton

from constants import SELECTED_GENES

tryptag = TrypTag()

base_directory = r'E:\PycharmProjects\BA\dataset\raw'
models_dir = "kmeans_models"


def load_model(gene):
    model_name = f"{gene}_channel_1"
    matching_models = [model for model in os.listdir(models_dir) if model_name in model]

    if not matching_models:
        raise FileNotFoundError

    model_name = matching_models[0]
    model_path = os.path.join(models_dir, model_name)
    kmeans = load(model_path)
    print(f"Loaded model {model_name} from {model_path}")
    return kmeans


def perform_kmeans_clustering(img, model):
    flattened_img = img.reshape(-1, 1)
    labels = model.predict(flattened_img)
    unique_labels = np.unique(labels)

    cluster_counts = [np.sum(labels == cluster_label) for cluster_label in unique_labels]
    smaller_cluster_label = unique_labels[np.argmin(cluster_counts)]
    cluster_mask = (labels == smaller_cluster_label).reshape(img.shape).astype(np.uint8)

    return cluster_mask


def get_largest_overlapping_component(mng_mask, phase_mask):
    # Find connected components in mng_mask
    retval, labels = cv2.connectedComponents(mng_mask.astype('uint8'))

    max_overlap = 0
    best_component = None

    for i in range(1, retval):
        # Create a binary mask for the current component
        component_mask = (labels == i).astype('uint8')

        # Calculate the overlap between the current component and the phase_mask
        overlap = cv2.bitwise_and(component_mask, phase_mask.astype('uint8'))
        overlap_sum = np.sum(overlap)

        # If the current component has the greatest overlap so far, update the best component and max overlap
        if overlap_sum > max_overlap:
            max_overlap = overlap_sum
            best_component = component_mask

    return best_component


def morphology_analysis(img):
    # Apply the second medial axis transform to the resized component
    pth_skeleton, distances = _mask_pruned_skeleton(img, 0, 15)
    neighbours = scipy.ndimage.convolve(pth_skeleton, [[1, 1, 1], [1, 0, 1], [1, 1, 1]]) * pth_skeleton
    termini_count = np.count_nonzero(neighbours == 1)
    midline_count = np.count_nonzero(neighbours == 2)
    branches_count = np.count_nonzero(neighbours > 2)
    morphology = {
        "termini": termini_count,
        "midlines": midline_count,
        "branches": branches_count,
    }
    # trace, if a single line (two termini, zero branches)
    if termini_count == 2 and branches_count == 0:
        termini = neighbours.copy()
        termini[termini > 1] = 0
        termini_y, termini_x = skimage.morphology.local_maxima(termini, indices=True, allow_borders=False)
        # trace from index 0
        midline = [[termini_y[0], termini_x[0]]]
        v = pth_skeleton[midline[-1][0], midline[-1][1]]
        while v > 0:
            v = 0
            # mark visited pixels by setting to 0
            pth_skeleton[midline[-1][0], midline[-1][1]] = 0
            # for all neighbours...
            for a in range(-1, 2):  # a is delta in x
                for b in range(-1, 2):  # b is delta in y
                    # if a skeleton pixel, step in that direction
                    if pth_skeleton[midline[-1][0] + b, midline[-1][1] + a] == 1:
                        midline.append([midline[-1][0] + b, midline[-1][1] + a])
                        v = pth_skeleton[midline[-1][0], midline[-1][1]]
                        # break inner loop on match
                        break
                # break outer loop with inner
                else:
                    continue
                break
        morphology.update({
            "midline": midline,
            "midline_pixels": len(midline),
            "distances": distances
        })

    return morphology


def calculate_midline_length(midline_analysis):
    # calculate distance along midline
    distance = [0]
    root2 = 2 ** 0.5
    for i in range(1, len(midline_analysis["midline"])):
        # step length 1 for orthogonal adjacent, root2 for diagonal adjacent
        if abs(midline_analysis["midline"][i][0] - midline_analysis["midline"][i - 1][0]) == 1 and abs(
                midline_analysis["midline"][i][1] - midline_analysis["midline"][i - 1][1]) == 1:
            distance.append(distance[-1] + root2)
        else:
            distance.append(distance[-1] + 1)

    return distance[-1]


@ray.remote
def process_gene(gene, terminus):
    kn_lengths = {}  # Dictionary to hold lengths and corrected lengths for each kn count
    model = load_model(gene)
    cell_line = CellLine(gene, terminus, "procyclic")
    rows = []
    for field_index in range(20):
        for cell_index in range(800):
            try:
                cell_image = tryptag.open_cell(cell_line, field_index, cell_index)
                mng_channel = cell_image.mng
                mng_mask = perform_kmeans_clustering(mng_channel, model)
                largest_component_mask = get_largest_overlapping_component(mng_mask,
                                                                           cell_image.phase_mask)  # Create a 3-channel image to overlay the largest component mask and the phase mask with different colors
                if largest_component_mask is None:
                    continue

                morphology = cell_morphology_analysis(cell_image)
                kn_count = morphology["count_kn"]

                flagella_morphology = morphology_analysis(largest_component_mask)

                if "midline" in flagella_morphology:
                    start = flagella_morphology["midline"][0]
                    end = flagella_morphology["midline"][-1]
                    flagella_length = calculate_midline_length(flagella_morphology)
                    corrected_flagella_length = (
                            flagella_morphology["distances"][start[0], start[1]] + flagella_morphology["distances"][
                        end[0], end[1]] + flagella_length)
                else:
                    continue

                if kn_count not in kn_lengths:
                    kn_lengths[kn_count] = {
                        'lengths': [],
                        'corrected_lengths': [],
                        'flagella_lengths': [],
                        'corrected_flagella_lengths': []
                    }

                skeleton, distance = skimage.morphology.medial_axis(cell_image.phase_mask, return_distance=True)
                start = morphology["anterior"]
                end = morphology["posterior"]
                length = morphology["length"]
                distance_ant = distance[start[0], start[1]]
                distance_post = distance[end[0], end[1]]
                corrected_length = (distance_ant + distance_post + length)

                kn_lengths[kn_count]['lengths'].append(length * tryptag.um_per_px)
                kn_lengths[kn_count]['corrected_lengths'].append(corrected_length * tryptag.um_per_px)
                kn_lengths[kn_count]['flagella_lengths'].append(flagella_length * tryptag.um_per_px)
                kn_lengths[kn_count]['corrected_flagella_lengths'].append(corrected_flagella_length * tryptag.um_per_px)

                signal_analysis = cell_signal_analysis(cell_image)

                try:
                    features = {
                        'count_kn': morphology['count_kn'],
                        'corrected_length': corrected_length * tryptag.um_per_px,
                        'corrected_flagella_length': corrected_flagella_length * tryptag.um_per_px,
                        'length': morphology['length'],
                        'count_k': morphology['count_k'],
                        'count_n': morphology['count_n'],
                        'cell_area': signal_analysis['cell_area'],
                        'mng_mean': signal_analysis['mng_mean'],
                        'mng_sum': signal_analysis['mng_sum'],
                        'mng_max': signal_analysis['mng_max'],
                        'midline_pixels': morphology['midline_pixels'],
                    }

                    for i in range(len(morphology['objects_k'])):
                        features.update({
                            f'area_k_{i}': morphology['objects_k'][i]['area'],
                            f'dna_sum_k_{i}': morphology['objects_k'][i]['dna_sum'],
                            f'dna_max_k_{i}': morphology['objects_k'][i]['dna_max'],
                            f'midline_index_k_{i}': morphology['objects_k'][i]['midline_index']
                        })

                    for i in range(len(morphology['objects_n'])):
                        features.update({
                            f'area_n_{i}': morphology['objects_n'][i]['area'],
                            f'dna_sum_n_{i}': morphology['objects_n'][i]['dna_sum'],
                            f'dna_max_n_{i}': morphology['objects_n'][i]['dna_max'],
                            f'midline_index_n_{i}': morphology['objects_n'][i]['midline_index']
                        })

                    # check if image already exists
                    if not os.path.exists(f"assets/{gene}_{field_index}_{cell_index}_phase.png"):
                        plt.imsave(f"assets/{gene}_{field_index}_{cell_index}_phase.png", cell_image.phase)
                        plt.imsave(f"assets/{gene}_{field_index}_{cell_index}_dna.png", cell_image.dna)

                    features["IMG_URL_PHASE"] = f"/assets/{gene}_{field_index}_{cell_index}_phase.png"
                    features["IMG_URL_DNA"] = f"/assets/{gene}_{field_index}_{cell_index}_dna.png"

                    rows.append(features)
                    print(f"File: {gene}, Field: {field_index}, Cell: {cell_index}, Count: {len(rows)}")

                except:
                    print("Failed to process cell (inner)")
            except:
                print("Failed to process cell (outer)")

    return rows


selected_genes = SELECTED_GENES

futures = [process_gene.remote(gene, selected_genes[gene][0]) for gene in selected_genes.keys()]
rows = ray.get(futures)

# Flatten the list of lists
rows = [item for sublist in rows for item in sublist]

df = pd.DataFrame(rows)
df = df.fillna(0)

df["norm_pos_k_0"] = df["midline_index_k_0"] / df["midline_pixels"]
df["norm_pos_n_0"] = df["midline_index_n_0"] / df["midline_pixels"]

pca = PCA(n_components=2)

# Create a scaler object
sc = StandardScaler()

# Select only the desired columns
df_selected = df[
    ['cell_area', 'corrected_length', 'corrected_flagella_length', 'area_k_0', 'area_n_0', 'dna_sum_k_0', 'dna_sum_n_0',
     'norm_pos_k_0', 'norm_pos_n_0']]

# Standardize the data
df_normalized = pd.DataFrame(sc.fit_transform(df_selected), columns=df_selected.columns)

# Apply PCA
df_pca = pd.DataFrame(pca.fit_transform(df_normalized), columns=['PCA1', 'PCA2'])

df['count_kn_numeric'] = pd.factorize(df['count_kn'])[0]

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
df_umap = pd.DataFrame(umap_model.fit_transform(df_normalized), columns=['UMAP1', 'UMAP2'])

# t-SNE
tsne_model = TSNE(n_components=2, random_state=42)
df_tsne = pd.DataFrame(tsne_model.fit_transform(df_normalized), columns=['TSNE1', 'TSNE2'])

df['UMAP1'] = df_umap['UMAP1']
df['UMAP2'] = df_umap['UMAP2']
df['TSNE1'] = df_tsne['TSNE1']
df['TSNE2'] = df_tsne['TSNE2']

# Pair plot
fig = px.scatter_matrix(df,
                        dimensions=['cell_area', 'corrected_length', 'corrected_flagella_length', 'area_k_0',
                                    'area_n_0', 'dna_sum_k_0', 'dna_sum_n_0',
                                    'norm_pos_k_0', 'norm_pos_n_0'],
                        color='count_kn', title='Scatter Matrix', labels={
        'cell_area': 'Area',
        'corrected_length': 'Length',
        'corrected_flagella_length': 'Flagella Length',
        'area_k_0': 'Area K',
        'area_n_0': 'Area N',
        'dna_sum_k_0': 'DNA K',
        'dna_sum_n_0': 'DNA N',
        'norm_pos_k_0': 'Norm Pos K',
        'norm_pos_n_0': 'Norm Pos N',
    })
fig.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig.update_traces(hoverinfo="none")

# UMAP plot
fig_umap = px.scatter(df,
                      x='UMAP1',
                      y='UMAP2',
                      color='count_kn',
                      title='UMAP Plot')

fig_umap.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig_umap.update_traces(hoverinfo="none", hovertemplate=None)

# t-SNE plot
fig_tsne = px.scatter(df,
                      x='TSNE1',
                      y='TSNE2',
                      color='count_kn',
                      title='t-SNE Plot')

fig_tsne.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig_tsne.update_traces(hoverinfo="none", hovertemplate=None)

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
    dcc.Graph(id="graph_umap", figure=fig_umap, clear_on_unhover=True),
    dcc.Graph(id="graph_tsne", figure=fig_tsne, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    [Input("graph_umap", "hoverData"),
     Input("graph_tsne", "hoverData")],
)
def display_hover(hoverData_umap, hoverData_tsne):
    hoverData = hoverData_umap if hoverData_umap is not None else hoverData_tsne
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src_phase = df_row['IMG_URL_PHASE']
    img_src_dna = df_row['IMG_URL_DNA']

    children = [
        html.Div(children=[
            html.Img(src=img_src_phase, style={"width": "100%"}),
            html.Img(src=img_src_dna, style={"width": "100%"}),
        ],
            style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=False)
