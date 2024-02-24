import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy
import skimage
from joblib import load
from matplotlib import patches
from tryptag import TrypTag, CellLine, tryptools
from tryptag.tryptools.tryptools import cell_morphology_analysis, _mask_pruned_skeleton

from constants import SELECTED_GENES, MODELS_DIR


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
    kn_lengths = {}  # Dictionary to hold lengths and corrected lengths for each kn count
    model = load_model(gene)
    cell_line = CellLine(gene, terminus, "procyclic")
    for field_index in range(20):
        for cell_index in range(800):
            try:
                plt.figure()
                cell_image = tryptag.open_cell(cell_line, field_index, cell_index)
                mng_channel = cell_image.mng
                mng_mask = perform_kmeans_clustering(mng_channel, model)
                largest_component_mask = get_largest_overlapping_component(mng_mask,
                                                                           cell_image.phase_mask)  # Create a 3-channel image to overlay the largest component mask and the phase mask with different colors
                if largest_component_mask is None:
                    continue

                morphology = cell_morphology_analysis(cell_image)
                kn_count = morphology["count_kn"]

                tryptools_plot = tryptools.plot_morphology_analysis(cell_image, morphology)

                # Ellipses corresponding to the major and minor axes
                for obj in morphology['objects_k'] + morphology['objects_n']:
                    color = 'red' if obj['type'] == 'k' else 'green'
                    ellipse = patches.Ellipse((obj["centroid"]["y"], obj["centroid"]["x"]),
                                              obj["major_axis_length"],
                                              obj["minor_axis_length"],
                                              angle=-obj["orientation"] * (180 / np.pi) + 90,
                                              fill=False,
                                              edgecolor=color,
                                              label=obj['type'] if 'k' in obj['type'] or 'n' in obj['type'] else None)
                    tryptools_plot.gca().add_patch(ellipse)

                # Adding labels, title, and legend
                tryptools_plot.xlabel('Position in px')
                tryptools_plot.ylabel('Position in px')
                tryptools_plot.title('Cell Morphology Analysis with Ellipses')

                handles, labels = tryptools_plot.gca().get_legend_handles_labels()
                anterior_legend = patches.Patch(color=(1.0, 0.0, 1.0, 1.0), label='Anterior')
                posterior_legend = patches.Patch(color=(1.0, 1.0, 0.0, 1.0), label='Posterior')

                handles.extend([anterior_legend, posterior_legend])
                labels.extend(["Anterior", "Posterior"])
                by_label = dict(zip(labels, handles))
                tryptools_plot.legend(by_label.values(), by_label.keys(), loc='upper right')

                # Masking zero values in the phase_mask
                masked_phase_mask = np.ma.masked_equal(cell_image.phase_mask, 0)

                # Masking zero values in the largest_component_mask
                masked_largest_component_mask = np.ma.masked_equal(largest_component_mask, 0)

                phase_max_value = np.max(cell_image.phase_mask)
                largest_component_max_value = np.max(largest_component_mask)

                # Display the phase mask in green with a specific alpha value (e.g., 0.5)
                # Set vmin to a small value greater than zero and vmax to the maximum value in the phase mask
                tryptools_plot.imshow(masked_phase_mask, cmap='Greens', alpha=0.5, vmin=0.01, vmax=phase_max_value)

                # Overlay the largest component mask in red with a different alpha value (e.g., 0.5)
                # Set vmin to a small value greater than zero and vmax to the maximum value in the largest component mask
                tryptools_plot.imshow(masked_largest_component_mask, cmap='Reds', alpha=0.5, vmin=0.01,
                                      vmax=largest_component_max_value)

                tryptools_plot.title(
                    "Cell " + str(cell_index) + " in field " + str(field_index) + "\nKN-Configuration: " + kn_count)

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

                # Define the directory path
                dir_path = os.path.join("flagella", str(kn_count))

                # Check if the directory exists, if not, create it
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # Define the file name based on cell_line, field_index, and cell_index
                file_name = f"{cell_line.gene_id}_{field_index}_{cell_index}.png"

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

                # Save the plot to the specified directory with the specified file name
                tryptools_plot.savefig(os.path.join(dir_path, file_name))
                plt.close()
            except:
                plt.close()

    return kn_lengths


def plot_histograms(kn_data):
    all_kn_lengths = kn_data
    font_size = 14
    plt.rcParams.update({'font.size': font_size})

    for kn, data in all_kn_lengths.items():
        plt.figure(figsize=(16, 7))

        num_samples = len(data['lengths'])  # Number of samples for each kn

        mean_length = np.mean(data['lengths'])
        std_length = np.std(data['lengths'])

        mean_corrected_length = np.mean(data['corrected_lengths'])
        std_corrected_length = np.std(data['corrected_lengths'])

        plt.subplot(1, 2, 1)
        plt.hist(data['lengths'], bins=50, alpha=0.7, label='Lengths', color='blue')
        plt.title(
            f'Lengths Histogram for {kn}\nMean: {mean_length:.2f} µm | Std: {std_length:.2f} µm | Samples: {num_samples}')
        plt.xlabel('Cell Length (µm)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")

        plt.subplot(1, 2, 2)
        plt.hist(data['corrected_lengths'], bins=50, alpha=0.7, label='Corrected Lengths', color='r')
        plt.title(
            f'Corrected Lengths Histogram for {kn}\nMean: {mean_corrected_length:.2f} µm | Std: {std_corrected_length:.2f} µm | Samples: {num_samples}')
        plt.xlabel('Corrected Cell Length (µm)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(f'{kn}_lengths.png')
        plt.show()

    for kn, data in all_kn_lengths.items():
        plt.figure(figsize=(16, 7))

        num_samples = len(data['flagella_lengths'])  # Number of samples for each kn

        mean_length = np.mean(data['flagella_lengths'])
        std_length = np.std(data['flagella_lengths'])

        mean_corrected_length = np.mean(data['corrected_flagella_lengths'])
        std_corrected_length = np.std(data['corrected_flagella_lengths'])

        plt.subplot(1, 2, 1)
        plt.hist(data['flagella_lengths'], bins=50, alpha=0.7, label='Lengths', color='blue')
        plt.title(
            f'Flagella Lengths Histogram for {kn}\nMean: {mean_length:.2f} µm | Std: {std_length:.2f} µm | Samples: {num_samples}')
        plt.xlabel('Flagella Length (µm)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")

        plt.subplot(1, 2, 2)
        plt.hist(data['corrected_flagella_lengths'], bins=50, alpha=0.7, label='Corrected Lengths', color='r')
        plt.title(
            f'Flagella Corrected Lengths Histogram for {kn}\nMean: {mean_corrected_length:.2f} µm | Std: {std_corrected_length:.2f} µm | Samples: {num_samples}')
        plt.xlabel('Flagella Corrected Cell Length (µm)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(f'{kn}_flagella_lengths.png')
        plt.show()


if __name__ == "__main__":
    """
    It will first create all the images and save them under ./flagella then it will calculate the statistics and save the images in the base directory for each KN configuration.
    Don't mind the RuntimeWarning from matplotlib, this is an error due to the multiprocessing.
    
    The actual statistics might vary by a few samples due to the randomness of the algorithm.
    """

    ray.init()
    tryptag = TrypTag()
    all_kn_lengths = {}
    futures = [process_gene.remote(gene, SELECTED_GENES[gene][0]) for gene in SELECTED_GENES.keys()]
    results = ray.get(futures)
    for kn_lengths in results:
        for kn, data in kn_lengths.items():
            if kn not in all_kn_lengths:
                all_kn_lengths[kn] = {
                    'lengths': [],
                    'corrected_lengths': [],
                    'flagella_lengths': [],
                    'corrected_flagella_lengths': []
                }
            all_kn_lengths[kn]['lengths'].extend(data['lengths'])
            all_kn_lengths[kn]['corrected_lengths'].extend(data['corrected_lengths'])
            all_kn_lengths[kn]['flagella_lengths'].extend(data['flagella_lengths'])
            all_kn_lengths[kn]['corrected_flagella_lengths'].extend(data['corrected_flagella_lengths'])

    plot_histograms(all_kn_lengths)
