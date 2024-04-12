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
from scipy import stats

import intel_sklearn_patch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tryptag import TrypTag, CellLine
from tryptag.tryptools import cell_signal_analysis
from tryptag.tryptools.tryptools import cell_morphology_analysis, _mask_pruned_skeleton

from constants import SELECTED_GENES, MODELS_DIR

tryptag = TrypTag()

os.makedirs("assets", exist_ok=True)


def load_model(gene):
    model_name = f"{gene}_channel_1"
    matching_models = [model for model in os.listdir(MODELS_DIR) if model_name in model]

    if not matching_models:
        raise FileNotFoundError

    model_name = matching_models[0]
    model_path = os.path.join(MODELS_DIR, model_name)
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


### THIS FUNCTION WAS COPIED FROM THE TRYPTAG PACKAGE AND POSSIBLY MODIFIED FOR MY PURPOSES ###
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


### THIS FUNCTION WAS COPIED FROM THE TRYPTAG PACKAGE AND POSSIBLY MODIFIED FOR MY PURPOSES ###
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

                skeleton, distance = skimage.morphology.medial_axis(cell_image.phase_mask, return_distance=True)
                start = morphology["anterior"]
                end = morphology["posterior"]
                length = morphology["length"]
                distance_ant = distance[start[0], start[1]]
                distance_post = distance[end[0], end[1]]
                corrected_length = (distance_ant + distance_post + length)

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

                    # add "minor_axis_length", "major_axis_length", "orientation"

                    for i in range(len(morphology['objects_k'])):
                        features.update({
                            f'area_k_{i}': morphology['objects_k'][i]['area'],
                            f'dna_sum_k_{i}': morphology['objects_k'][i]['dna_sum'],
                            f'dna_max_k_{i}': morphology['objects_k'][i]['dna_max'],
                            f'midline_index_k_{i}': morphology['objects_k'][i]['midline_index'],
                            f'minor_axis_length_k_{i}': morphology['objects_k'][i]['minor_axis_length'],
                            f'major_axis_length_k_{i}': morphology['objects_k'][i]['major_axis_length'],
                            f'orientation_k_{i}': morphology['objects_k'][i]['orientation']
                        })

                    for i in range(len(morphology['objects_n'])):
                        features.update({
                            f'area_n_{i}': morphology['objects_n'][i]['area'],
                            f'dna_sum_n_{i}': morphology['objects_n'][i]['dna_sum'],
                            f'dna_max_n_{i}': morphology['objects_n'][i]['dna_max'],
                            f'midline_index_n_{i}': morphology['objects_n'][i]['midline_index'],
                            f'minor_axis_length_n_{i}': morphology['objects_n'][i]['minor_axis_length'],
                            f'major_axis_length_n_{i}': morphology['objects_n'][i]['major_axis_length'],
                            f'orientation_n_{i}': morphology['objects_n'][i]['orientation']
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

futures = [process_gene.remote(gene, selected_genes[gene][0]) for gene in list(selected_genes.keys())]
rows = ray.get(futures)

# Flatten the list of lists
rows = [item for sublist in rows for item in sublist]

df = pd.DataFrame(rows)
df = df.fillna(0)

df["norm_pos_k_0"] = df["midline_index_k_0"] / df["midline_pixels"]
df["norm_pos_n_0"] = df["midline_index_n_0"] / df["midline_pixels"]

# Create a scaler object
sc = StandardScaler()

features = ['cell_area', 'corrected_length', 'corrected_flagella_length', 'area_k_0',
            'area_n_0', 'dna_sum_k_0', 'dna_sum_n_0', 'norm_pos_k_0', 'norm_pos_n_0',
            'major_axis_length_k_0', 'major_axis_length_n_0']

# Select only the desired columns
df_selected = df[features]

# Filtering for specific cell types
filtered_df = df[df['count_kn'].isin(['1K1N', '2K1N', '2K2N'])]

# List to hold results
results = []

for i, feature_x in enumerate(features):
    for feature_y in features[i + 1:]:
        # Extracting the values for each feature
        x = filtered_df[feature_x].values
        y = filtered_df[feature_y].values

        # Calculate the best fit line (linear regression)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Storing results
        results.append({
            'Pair': (feature_x, feature_y),
            'Pearson Correlation Coefficient': r_value,
            'P-Value': p_value,
            'Standard Error': std_err
        })

# Sorting the results by the correlation coefficient in descending order
sorted_results = sorted(results, key=lambda x: x['Pearson Correlation Coefficient'], reverse=True)

# Printing the sorted results
for result in sorted_results:
    print(f"Pair: {result['Pair']}")
    print(f"  Pearson Correlation Coefficient: {result['Pearson Correlation Coefficient']}")
    print(f"  P-Value: {result['P-Value']}")
    print(f"  Standard Error: {result['Standard Error']}\n")

# Standardize the data
df_normalized = pd.DataFrame(sc.fit_transform(df_selected), columns=df_selected.columns)

# PCA
pca_model = PCA(n_components=2, random_state=42)
df_pca = pd.DataFrame(pca_model.fit_transform(df_normalized), columns=['PCA1', 'PCA2'])

print(pca_model.components_)

# Feature importance for PCA can be derived from the absolute values of the loading scores
feature_importance_pca = np.abs(pca_model.components_)

contribution_scores = np.square(feature_importance_pca)

# Normalizing the contribution scores to sum to 1 for each component
contribution_percentage = contribution_scores / np.sum(contribution_scores, axis=1, keepdims=True)

# Converting to percentage
contribution_percentage *= 100

# Creating a DataFrame for a nicer presentation
df_contribution_percentage = pd.DataFrame(contribution_percentage, columns=df_normalized.columns,
                                          index=['PCA1', 'PCA2']).transpose()

# Sorting by contribution to PCA1 for presentation
pca1_sorted = df_contribution_percentage['PCA1'].sort_values(ascending=False)
pca2_sorted = df_contribution_percentage['PCA2'].sort_values(ascending=False)

# Creating a new DataFrame for a nicely formatted presentation
df_sorted_contributions = pd.DataFrame({
    'Feature_PCA1': pca1_sorted.index,
    'PCA1_Contribution%': pca1_sorted.values,
    'Feature_PCA2': pca2_sorted.index,
    'PCA2_Contribution%': pca2_sorted.values
})

df_sorted_contributions.reset_index(drop=True, inplace=True)

print(df_sorted_contributions.to_string())

print("Variance explained by each principal component:")
for i, variance in enumerate(pca_model.explained_variance_ratio_):
    print(f"PC{i+1}: {variance*100:.2f}%")

# Optionally, to see the cumulative variance explained:
cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
print("\nCumulative variance explained by the first n components:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"First {i+1} PC(s): {cum_var*100:.2f}%")
# t-SNE
tsne_model = TSNE(n_components=2, random_state=42)
df_tsne = pd.DataFrame(tsne_model.fit_transform(df_normalized), columns=['TSNE1', 'TSNE2'])

# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
df_umap = pd.DataFrame(umap_model.fit_transform(df_normalized), columns=['UMAP1', 'UMAP2'])

df['PCA1'] = df_pca['PCA1']
df['PCA2'] = df_pca['PCA2']
df['TSNE1'] = df_tsne['TSNE1']
df['TSNE2'] = df_tsne['TSNE2']
df['UMAP1'] = df_umap['UMAP1']
df['UMAP2'] = df_umap['UMAP2']

# Pair plot
fig = px.scatter_matrix(df,
                        dimensions=['cell_area', 'corrected_length', 'corrected_flagella_length', 'area_k_0',
                                    'area_n_0', 'dna_sum_k_0', 'dna_sum_n_0',
                                    'norm_pos_k_0', 'norm_pos_n_0', 'major_axis_length_k_0', 'major_axis_length_n_0'],
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
        'major_axis_length_k_0': 'Major Axis K',
        'major_axis_length_n_0': 'Major Axis N',
    })
fig.update_layout(legend_title_text='Legend', width=2000, height=2000)
fig.update_traces(hoverinfo="none")

# PCA plot
fig_pca = px.scatter(df,
                     x='PCA1',
                     y='PCA2',
                     color='count_kn',  # Ensure this matches the color mapping used in other plots
                     title='PCA Plot')

fig_pca.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig_pca.update_traces(hoverinfo="none", hovertemplate=None)

# t-SNE plot
fig_tsne = px.scatter(df,
                      x='TSNE1',
                      y='TSNE2',
                      color='count_kn',
                      title='t-SNE Plot')

fig_tsne.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig_tsne.update_traces(hoverinfo="none", hovertemplate=None)

# UMAP plot
fig_umap = px.scatter(df,
                      x='UMAP1',
                      y='UMAP2',
                      color='count_kn',
                      title='UMAP Plot')

fig_umap.update_layout(legend_title_text='Legend', width=1000, height=1000)
fig_umap.update_traces(hoverinfo="none", hovertemplate=None)

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),  # Scatter Matrix
    dcc.Graph(id="graph_pca", figure=fig_pca, clear_on_unhover=True),  # PCA
    dcc.Graph(id="graph_tsne", figure=fig_tsne, clear_on_unhover=True),  # t-SNE
    dcc.Graph(id="graph_umap", figure=fig_umap, clear_on_unhover=True),  # UMAP
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    [Input("graph_umap", "hoverData"),
     Input("graph_tsne", "hoverData"),
     Input("graph_pca", "hoverData")],  # Include PCA in the callback
)
def display_hover(hoverData_umap, hoverData_tsne, hoverData_pca):
    # Choose the hover data to display based on which plot is hovered over
    hoverData = hoverData_umap if hoverData_umap is not None else hoverData_tsne if hoverData_tsne is not None else hoverData_pca
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
