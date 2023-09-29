from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.widgets import RangeSlider
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans


def plot_channels(image, num_channels):
    fig, axs = plt.subplots(1, num_channels, figsize=(7 * num_channels, 5))
    for i in range(num_channels):
        channel = image[i, :, :]
        axs[i].imshow(channel, cmap='viridis')
        axs[i].set_title(f'Channel {i + 1}')
    plt.tight_layout()
    plt.show()


def plot_clusters(image, num_channels, clusters):
    fig, axs = plt.subplots(2, num_channels, figsize=(7 * num_channels, 10))
    kmeans_models = []
    clusterings = []
    for i in range(num_channels):
        channel = image[i, :, :]
        reshaped_channel = channel.reshape(-1, 1)

        kmeans_model_path = Path("kmeans_models") / f"kmeans_model_channel_{i}_{clusters[i]}.joblib"
        print(f"Performing clustering and saving KMeans model to {kmeans_model_path}")
        kmeans = KMeans(n_clusters=clusters[i], random_state=0).fit(reshaped_channel)

        kmeans_models.append(kmeans)

        clustered_channel = kmeans.labels_.reshape(channel.shape)
        clusterings.append(clustered_channel)

        axs[0, i].imshow(clustered_channel, cmap='viridis')
        axs[0, i].set_title(f'Clustered Channel {i + 1}')
        unique_labels, counts = np.unique(clustered_channel, return_counts=True)
        colors = plt.cm.viridis(unique_labels / clusters[i])
        axs[1, i].bar(unique_labels, counts, color=colors)
        axs[1, i].set_xticks(unique_labels)
        axs[1, i].set_title(f'Histogram of Clustered Channel {i + 1}')
    plt.tight_layout()
    plt.show()
    return kmeans_models, clusterings


def plot_clusters_binary(clusterings, clusters):
    max_clusters = max(clusters)
    num_channels = len(clusterings)
    fig, axs = plt.subplots(max_clusters, num_channels, figsize=(7 * num_channels, 20 * max_clusters / 3))
    for i in range(num_channels):
        for j in range(clusters[i]):
            binary_image = np.where(clusterings[i] == j, 1, 0)
            axs[j, i].imshow(binary_image, cmap='gray')
            axs[j, i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_cluster_islands(clusterings):
    binary_image = np.where(clusterings[0] == 1, 1, 0)
    labeled_image, num = label(binary_image, return_num=True, connectivity=2)

    # Create a colormap that maps 0 to black and other labels to colors from viridis
    cmap = plt.cm.viridis
    # create a new colormap from the existing colormap
    colors = cmap(np.linspace(0, 1, num + 1))
    colors[0] = [0, 0, 0, 1]  # set the color for label 0 to black
    new_cmap = matplotlib.colors.ListedColormap(colors)

    plt.imshow(labeled_image, cmap=new_cmap)
    plt.title(f'There are {num} islands in the binary image')

    # Compute the centroid for each labeled region
    properties = regionprops(labeled_image)
    for prop in properties:
        # prop.label gives the label number
        # prop.centroid returns the centroid of the labeled region
        y0, x0 = prop.centroid
        plt.plot(x0, y0, '.r')  # plot red dot at the center
        plt.text(x0, y0, str(prop.area), color='red')  # add the number of the label

    plt.show()


def plot_area_distribution(clusterings):
    binary_image = np.where(clusterings[0] == 1, 1, 0)
    labeled_image, num = label(binary_image, return_num=True, connectivity=2)

    # Compute the area for each labeled region
    properties = regionprops(labeled_image)
    areas = [prop.area for prop in properties]

    # Plot histogram
    plt.hist(areas, bins=25, edgecolor='black')
    plt.title('Area distribution of labeled regions')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    plt.show()


def plot_cluster_islands_interactive(clusterings):
    binary_image = np.where(clusterings[0] == 1, 1, 0)
    labeled_image, num = label(binary_image, return_num=True, connectivity=2)

    properties = regionprops(labeled_image)
    areas = np.array([prop.area for prop in properties])

    # Setup figure with three subplots, giving more space to the image subplot
    fig, (ax_img, ax_hist, ax_slider) = plt.subplots(3, 1, figsize=(7, 15),
                                                     gridspec_kw={'height_ratios': [5, 2, 1]})

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, num + 1))
    colors[0] = [0, 0, 0, 1]
    new_cmap = matplotlib.colors.ListedColormap(colors)
    im = ax_img.imshow(labeled_image, cmap=new_cmap)
    title = ax_img.set_title(f'There are {num} islands in the binary image')

    ax_hist.hist(areas, bins=50, edgecolor='black')
    ax_hist.set_title('Area distribution of labeled regions')
    ax_hist.set_xlabel('Area (pixels)')
    ax_hist.set_ylabel('Frequency')

    area_min, area_max = int(areas.min()), int(areas.max())
    slider_range = RangeSlider(ax_slider, "Area Threshold", area_min, area_max, valinit=(area_min, area_max), valstep=5)

    masks = {i: np.isin(labeled_image, np.where(areas == i)[0] + 1)
             for i in np.unique(areas)}

    def update(val):
        min_val, max_val = int(val[0]), int(val[1])
        filtered_image = np.zeros_like(labeled_image)
        current_num_islands = 0
        for i in range(min_val, max_val + 1):
            if i in masks:
                current_num_islands += np.sum(masks[i])
                filtered_image[masks[i]] = labeled_image[masks[i]]
        title.set_text(f'There are {current_num_islands} islands in the binary image')
        im.set_data(filtered_image)
        fig.canvas.draw_idle()

    slider_range.on_changed(update)

    plt.tight_layout()
    plt.show()


def main(file_path):
    # Ensure directories exist
    Path("kmeans_models").mkdir(parents=True, exist_ok=True)
    Path("clusterings").mkdir(parents=True, exist_ok=True)

    image = tiff.imread(file_path)
    num_channels = image.shape[0]
    clusters = [4, 2, 2]
    # plot_channels(image, num_channels)
    kmeans_models, clusterings = plot_clusters(image, num_channels, clusters)
    plot_clusters_binary(clusterings, clusters)
    plot_cluster_islands(clusterings)
    plot_area_distribution(clusterings)
    plot_cluster_islands_interactive(clusterings)


main(r'_tryptag_cache\A1693_20192507\Tb927.10.11300_4_N_5.tif')
