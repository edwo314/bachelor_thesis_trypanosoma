import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.ticker import MaxNLocator
from sklearnex import patch_sklearn

# sklearnex will use intel extensions for sklearn
patch_sklearn()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ray
from ray import get
from skimage.transform import resize

ray.init()


def function(image: str, channel: int, silhouette_range: range):
    img = tifffile.imread(image)[channel, :, :]

    # resizing the images is recommended for silhouette score
    # img = resize(img, (img.shape[0] // 4, img.shape[1] // 4))
    middle_x = img.shape[0] // 2
    middle_y = img.shape[1] // 2
    middle_size = 512

    img = img[middle_x - middle_size // 2 : middle_x + middle_size // 2,
                         middle_y - middle_size // 2 : middle_y + middle_size // 2]

    @ray.remote
    def compute_kmeans_and_scores_for_image(k: int):
        img_array = np.expand_dims(img, axis=0)
        pixel_array = img_array.reshape(-1, 1)

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixel_array)
        silhouette = silhouette_score(pixel_array, labels, random_state=42)  # calculate silhouette score
        clustered_image = labels.reshape(img_array.shape)
        inertia = kmeans.inertia_
        return inertia, silhouette, clustered_image,

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

    # Determine the figure size based on the number of subplots
    fig_width = 5 * (len(silhouette_range) + 1)
    fig_height = 5

    fig, axes = plt.subplots(1, (len(silhouette_range) + 1), figsize=(fig_width, fig_height))
    axes[0].imshow(img, aspect='auto', cmap="gray")
    axes[0].set_title(f'Base Image')
    axes[0].axis('off')

    for j, clustered_image in enumerate(clustered_images_list):
        K = silhouette_range[j]
        cmap = plt.get_cmap('nipy_spectral', K)
        axes[j+1].imshow(clustered_image[0], cmap=cmap, aspect='auto')
        axes[j+1].set_title(f'K={K}')
        axes[j+1].axis('off')

    fig.suptitle('Clustered Image with different K', fontsize=16)
    plt.tight_layout()  # Minimize whitespace between subplots
    plt.savefig(f'clustered_image_channel_{channel}.png')  # Save the figure
    plt.show()

    # Plotting the silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(silhouette_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-ticks are integers
    plt.grid(True)
    plt.savefig(f'silhouette_scores_channel_{channel}_full.png')  # Save the figure
    plt.show()

    # Plotting the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(silhouette_range, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-ticks are integers
    plt.grid(True)
    plt.savefig(f'elbow_method_channel_{channel}_full.png')  # Save the figure
    plt.show()


if __name__ == "__main__":
    # pairs is setting the number of clusters and the channel to be used
    pairs = ((4, 0), (2, 1), (2, 2))
    for k, c in pairs:
        function(r"dataset\raw\Tb927.3.4290\Tb927.3.4290_4_N_1.tif", c, range(k, k + 4))
