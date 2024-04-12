import matplotlib.pyplot as plt
import numpy as np
import ray
import tifffile
from cv2 import resize
from matplotlib.ticker import MaxNLocator
from ray import get

import intel_sklearn_patch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from constants import DATASET_DIR

ray.init()


def get_clustering_metrics(image: str, channel: int, silhouette_range: range, resize_factor: int = 4):
    img = tifffile.imread(image)[channel, :, :]

    # resizing the images is recommended for silhouette score
    img = resize(img, (img.shape[0] // resize_factor, img.shape[1] // resize_factor))
    middle_x = img.shape[0] // 2
    middle_y = img.shape[1] // 2
    middle_size = 512 // resize_factor

    img = img[middle_x - middle_size // 2: middle_x + middle_size // 2,
          middle_y - middle_size // 2: middle_y + middle_size // 2]

    @ray.remote
    def compute_kmeans_and_scores_for_image(k: int):
        img_array = np.expand_dims(img, axis=0)
        pixel_array = img_array.reshape(-1, 1)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(pixel_array)
        silhouette = silhouette_score(pixel_array, labels, random_state=42)  # calculate silhouette score
        clustered_image = labels.reshape(img_array.shape)
        inertia = kmeans.inertia_
        return inertia, silhouette, clustered_image

    # Start all tasks in parallel
    results = [compute_kmeans_and_scores_for_image.remote(k) for k in silhouette_range]

    # Fetch the results
    results = get(results)
    ray.shutdown()

    # Initialize lists to store silhouette scores and clustered images
    inertias = []
    silhouette_scores = []
    clustered_images_list = []

    for inertia, silhouette, clustered_image in results:
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        clustered_images_list.append(clustered_image)

    return inertias, silhouette_scores, clustered_images_list, img


def plot_clustering_results(inertias, silhouette_scores, clustered_images_list, img, channel, silhouette_range):
    # Plotting the clustered images
    fig_width = 5 * (len(silhouette_range) + 1)
    fig_height = 5

    fig, axes = plt.subplots(1, (len(silhouette_range) + 1), figsize=(fig_width, fig_height))
    axes[0].imshow(img, aspect='auto', cmap="gray")
    axes[0].set_title(f'Base Image')
    axes[0].axis('off')

    for j, clustered_image in enumerate(clustered_images_list):
        K = silhouette_range[j]
        cmap = plt.get_cmap('nipy_spectral', K)
        axes[j + 1].imshow(clustered_image[0], cmap=cmap, aspect='auto')
        axes[j + 1].set_title(f'K={K}')
        axes[j + 1].axis('off')

    fig.suptitle('Clustered Image with different K', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'clustered_image_channel_{channel}.png')
    plt.show()

    # Plotting the silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(silhouette_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.savefig(f'silhouette_scores_channel_{channel}_full.png')
    plt.show()

    # Plotting the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(silhouette_range, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.savefig(f'elbow_method_channel_{channel}_full.png')
    plt.show()


if __name__ == "__main__":
    """
    Generates plots for the Elbow Method and Silhouette Scores for a given input image to evaluate the optimal number of clusters.
    It saves 3 plots for each channel: elbow method, silhouette scores and clustered image.

    WARNING: Running it with resize factor 1 will take a very long time (~20 minutes), as the silhouette score calculation is very expensive.
    If that is too long for you to wait, you can set resize_factor to a larger integer like 4 or 8. However, as a consequence the image quality will decrease and the
    metrics won't be as accurate. Resize factor 4 will decrease the number of pixels by a factor of 16.
    """

    # pairs is a tuple of tuples with (number of clusters, channel)
    # 4 clusters for channel 0
    # 2 clusters for channel 1
    # 2 clusters for channel 2
    # e.g we expect a minimum of 4 clusters in the phase channel, it will calculate it for 4, 5, 6, 7 clusters, this is set by the silhouette range
    pairs = ((4, 0), (2, 1), (2, 2))
    for clusters, channel in pairs:
        inertias, silhouette_scores, clustered_images_list, img = get_clustering_metrics(
            rf"{DATASET_DIR}\Tb927.3.4290\Tb927.3.4290_4_N_1.tif", channel=channel,
            silhouette_range=range(clusters, clusters + 4), resize_factor=8)
        plot_clustering_results(inertias, silhouette_scores, clustered_images_list, img, channel,
                                range(clusters, clusters + 4))
