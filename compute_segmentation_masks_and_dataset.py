import os
import random
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import ray as ray
import tifffile as tiff
from joblib import load, dump
from matplotlib import pyplot as plt, patches
from scipy import ndimage
from sklearnex import patch_sklearn

from constants import SELECTED_GENES

patch_sklearn()

from sklearn.cluster import KMeans
from tqdm import tqdm


def fill_holes(mask, kernel_size=5):
    hole_filled_mask = ndimage.binary_fill_holes(mask)

    # Convert boolean array to uint8
    hole_filled_mask = (hole_filled_mask * 255).astype(np.uint8)

    # Closing operation to further smooth and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(hole_filled_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


def filter_islands_by_size(mask, min_size=0, max_size=10000):
    # Connected component analysis and size filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_8U)

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        if min_size < stats[i, cv2.CC_STAT_AREA] < max_size:
            filtered_mask[labels == i] = 255
    return filtered_mask


@dataclass
class Cell:
    index: int
    stats: tuple
    base_path: str = None

    @property
    def phase_img(self):
        img = cv2.imread(os.path.join(self.base_path, "phase.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    @property
    def dna_img(self):
        img = cv2.imread(os.path.join(self.base_path, "dna.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    @property
    def mng_img(self):
        img = cv2.imread(os.path.join(self.base_path, "mng.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    @property
    def phase_mask(self):
        img = cv2.imread(os.path.join(self.base_path, "phase_mask.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    @property
    def dna_mask(self):
        img = cv2.imread(os.path.join(self.base_path, "dna_mask.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    @property
    def mng_mask(self):
        img = cv2.imread(os.path.join(self.base_path, "mng_mask.png"), cv2.IMREAD_GRAYSCALE)
        return self.crop_image(img)

    def crop_image(self, img):
        x, y, w, h, _ = self.stats
        return img[y:y + h, x:x + w]


@dataclass
class Field:
    index: int
    terminus: str
    gene_name: str
    img_path: str
    mask_path: str
    base_path: str = None
    cells: List[Cell] = None

    def __post_init__(self):
        self.cells = []
        self.initialize_images()
        # self.get_cells()

    def get_cells(self):
        # Load the image from the base_path + phase_mask.png
        img_path = os.path.join(self.base_path, 'phase_mask.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Load the DNA mask for nucleus and kinetoplast analysis
        dna_mask_path = os.path.join(self.base_path, 'dna_mask.png')
        dna_mask = cv2.imread(dna_mask_path, cv2.IMREAD_GRAYSCALE)

        # Perform connected components analysis on DNA mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dna_mask, connectivity=8)

        # Filter out objects touching the border and outside size range
        height, width = dna_mask.shape
        min_size = 5
        max_size = 500
        filtered_stats = [stat for stat in stats[1:] if
                          stat[cv2.CC_STAT_LEFT] > 0 and
                          stat[cv2.CC_STAT_TOP] > 0 and
                          stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] < width and
                          stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] < height and
                          stat[cv2.CC_STAT_AREA] >= min_size and
                          stat[cv2.CC_STAT_AREA] <= max_size]

        # Calculate the major and minor axes lengths for ellipses and store data for clustering
        axes_data = []
        ellipses = []
        for i, stat in enumerate(filtered_stats):
            # Retrieve a binary mask for the current object
            mask = (labels == (i + 1)).astype(np.uint8)

            # Find contours and fit an ellipse
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours[0]) < 5:
                continue
            ellipse = cv2.fitEllipse(contours[0])
            (xc, yc), (d1, d2), angle = ellipse

            major_axis_length = d2
            minor_axis_length = d1
            ellipses.append(ellipse)
            axes_data.append([minor_axis_length, major_axis_length])

        minor_axes = [x[0] for x in axes_data]
        major_axes = [x[1] for x in axes_data]

        def remove_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return [x for x in data if lower_bound < x < upper_bound]

        # Remove outliers from minor and major axes
        minor_axes_clean = remove_outliers(minor_axes)
        major_axes_clean = remove_outliers(major_axes)

        # Perform 1D clustering on cleaned minor and major axes separately
        kmeans_minor = KMeans(n_clusters=2).fit(np.array(minor_axes_clean).reshape(-1, 1))
        kmeans_major = KMeans(n_clusters=2).fit(np.array(major_axes_clean).reshape(-1, 1))

        nuclei_label_minor = np.argmax(np.bincount(kmeans_minor.labels_))
        nuclei_label_major = np.argmax(np.bincount(kmeans_major.labels_))

        def plot_histogram(data, labels, nuclei_label, title):
            # Determine the number of bins and the range for the histogram
            bins = 50
            range_min, range_max = 0, 500
            bin_width = (range_max - range_min) / bins

            # Create the histogram bins manually
            hist, bin_edges = np.histogram(data, bins=bins, range=(range_min, range_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Plot each bin with the corresponding color
            for i in range(bins):
                # Determine the color for this bin based on the clustering labels
                color = 'red' if np.mean([labels[j] for j in range(len(data)) if
                                          bin_edges[i] <= data[j] < bin_edges[i + 1]]) == nuclei_label else 'blue'
                plt.bar(bin_centers[i], hist[i], width=bin_width, color=color, edgecolor='black')

            plt.title(title)
            plt.show()

        # Plot histograms for minor and major axes
        plot_histogram(minor_axes_clean, kmeans_minor.labels_, nuclei_label_minor, 'Minor Axes Clustering')
        plot_histogram(major_axes_clean, kmeans_major.labels_, nuclei_label_major, 'Major Axes Clustering')

        final_labels = []
        for minor_label, major_label in zip(kmeans_minor.labels_, kmeans_major.labels_):
            if minor_label == major_label:
                final_labels.append(minor_label)
            else:
                final_labels.append(major_label)

        # Map the final labels to nuclei and kinetoplast using the major axis labels
        final_labels = [nuclei_label_major if label == nuclei_label_major else 1 - nuclei_label_major for label in
                        final_labels]

        ## Perform clustering into two clusters: kinetoplasts and nuclei
        # kmeans = KMeans(n_clusters=2).fit(axes_data)
        # cluster_centers = kmeans.cluster_centers_
        # nuclei_cluster = np.argmax(cluster_centers.mean(axis=1))
        #
        ## Assign objects to kinetoplasts and nuclei and plot ellipses if required
        # kinetoplast_data = {'minor_axis_length': [], 'major_axis_length': []}
        # nuclei_data = {'minor_axis_length': [], 'major_axis_length': []}
        #
        # plt.imshow(dna_mask, cmap='gray')
        # Initialize plot with the DNA mask
        plt.figure(figsize=(10, 10), dpi=300)  # Adjust size as needed
        plt.imshow(dna_mask, cmap='gray')

        # Iterate through the final labels and the axes data
        for label, (minor_axis_length, major_axis_length), ellipse in zip(final_labels, axes_data,
                                                                          ellipses):
            (xc, yc), (d1, d2), angle = ellipse
            # Determine color (e.g., red for nuclei, blue for kinetoplast)
            color = 'red' if label == nuclei_label_major else 'blue'

            # Create an Ellipse patch object with the calculated parameters
            ellipse_patch = patches.Ellipse((xc, yc), major_axis_length, minor_axis_length, angle=angle + 90,
                                            fill=False, edgecolor=color)

            # Add the patch to the current plot
            plt.gca().add_patch(ellipse_patch)

        # Show the plot
        plt.show()

        # for i, label in enumerate(kmeans.labels_):
        #    data = {
        #        'minor_axis_length': axes_data[i][0],
        #        'major_axis_length': axes_data[i][1],
        #    }
        #    if label == nuclei_cluster:
        #        nuclei_data['minor_axis_length'].append(data['minor_axis_length'])
        #        nuclei_data['major_axis_length'].append(data['major_axis_length'])
        #    else:
        #        kinetoplast_data['minor_axis_length'].append(data['minor_axis_length'])
        #        kinetoplast_data['major_axis_length'].append(data['major_axis_length'])
        #
        #    (xc, yc), (d1, d2), angle = cv2.fitEllipse(contours[0])  # Store the tuple here
        #    major_axis_length = d1
        #    minor_axis_length = d2
        #
        #    obj_centroid = centroids[i + 1]
        #    ellipse = patches.Ellipse((obj_centroid[0], obj_centroid[1]),
        #                              major_axis_length,
        #                              minor_axis_length,
        #                              angle=angle,  # Corrected line
        #                              fill=True,
        #                              edgecolor='red' if label == nuclei_cluster else 'blue')
        #    plt.gca().add_patch(ellipse)
        # plt.show()

        # Check if image was properly loaded
        if img is None:
            raise FileNotFoundError(f"Could not load image from {img_path}")

        # Get all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

        # Throw out cells that are touching the border of the image
        height, width = img.shape
        border_labels = [i for i in range(num_labels) if
                         stats[i, cv2.CC_STAT_LEFT] == 0 or
                         stats[i, cv2.CC_STAT_TOP] == 0 or
                         stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] == width or
                         stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == height]

        # Define size range for cells
        min_size = 800  # Example value
        max_size = 5000  # Example value

        # For each component (excluding the background), add a cell to the cells list
        for i in range(1, num_labels):
            # Throw out cells that are touching the border
            if i in border_labels:
                continue

            # Throw out cells that are too small or too large
            if stats[i, cv2.CC_STAT_AREA] < min_size or stats[i, cv2.CC_STAT_AREA] > max_size:
                continue

            # Throw out cells that don't contain a nucleus and kinetoplast
            # You would need to implement specific logic for this condition.
            # Example:
            # if not contains_nucleus_and_kinetoplast(img, stats[i]):
            #     continue

            cell = Cell(index=i, stats=tuple(stats[i]), base_path=self.base_path)
            self.cells.append(cell)

    @property
    def phase_img(self):
        return cv2.imread(os.path.join(self.base_path, "phase.png"), cv2.IMREAD_GRAYSCALE)

    @property
    def dna_img(self):
        return cv2.imread(os.path.join(self.base_path, "dna.png"), cv2.IMREAD_GRAYSCALE)

    @property
    def mng_img(self):
        return cv2.imread(os.path.join(self.base_path, "mng.png"), cv2.IMREAD_GRAYSCALE)

    @property
    def phase_mask(self):
        return cv2.imread(os.path.join(self.base_path, "phase_mask.png"), cv2.IMREAD_GRAYSCALE)

    @property
    def dna_mask(self):
        return cv2.imread(os.path.join(self.base_path, "dna_mask.png"), cv2.IMREAD_GRAYSCALE)

    @property
    def mng_mask(self):
        return cv2.imread(os.path.join(self.base_path, "mng_mask.png"), cv2.IMREAD_GRAYSCALE)

    def initialize_images(self):
        img = self.load_image(self.img_path)
        phase_img, mng_img, dna_img = img[0, :, :], img[1, :, :], img[2, :, :]

        # Save each image as PNG using OpenCV
        cv2.imwrite(os.path.join(self.base_path, "phase.png"), phase_img)
        cv2.imwrite(os.path.join(self.base_path, "dna.png"), dna_img)
        cv2.imwrite(os.path.join(self.base_path, "mng.png"), mng_img)

        # Create and save masks
        self.create_and_save_mask(phase_img, os.path.join(self.base_path, "phase_mask.png"))
        self.create_and_save_mask(dna_img, os.path.join(self.base_path, "dna_mask.png"))
        self.create_and_save_mask(mng_img, os.path.join(self.base_path, "mng_mask.png"))

    def create_and_save_mask(self, img, mask_path):
        show_images = False
        if "phase" in mask_path:
            phase_mask = self.create_phase_mask(img, show_images=show_images)
            cv2.imwrite(mask_path, phase_mask)
        elif "dna" in mask_path:
            dna_mask = self.create_dna_mask(img, show_images=show_images)
            cv2.imwrite(mask_path, dna_mask)
        elif "mng" in mask_path:
            mng_mask = self.create_mng_mask(img, show_images=show_images)
            cv2.imwrite(mask_path, mng_mask)

    def load_image(self, img_path):
        return tiff.imread(img_path)

    def _get_model_name_and_path(self, channel):
        models_dir = "kmeans_models"
        basepath_key = self.base_path.split("\\")[-2] + "_channel_" + str(channel)
        model_files = os.listdir(models_dir)

        matching_models = [model for model in model_files if basepath_key in model]
        if not matching_models:
            raise FileNotFoundError(f"No model files found for {basepath_key} in {models_dir}.")

        # Select the first model
        model_name = matching_models[0]
        model_path = os.path.join(models_dir, model_name)

        return model_name, model_path

    def _perform_kmeans_clustering(self, img, channel, selected_cluster, show_images=False,
                                   select_smaller_cluster=False):
        model_name, model_path = self._get_model_name_and_path(channel)

        flattened_img = img.reshape(-1, 1)
        kmeans = load(model_path)
        labels = kmeans.predict(flattened_img)
        unique_labels = np.unique(labels)

        if show_images:
            num_clusters = len(unique_labels)
            num_cols = 3  # You can adjust this
            num_rows = max(num_clusters // num_cols, 1)
            if num_clusters % num_cols:
                num_rows += 1  # Add an extra row if the clusters cannot be arranged evenly

            fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 3))

            for i, label in enumerate(unique_labels):
                cluster_mask = (labels == label).reshape(img.shape)

                row_idx = i // num_cols
                col_idx = i % num_cols

                # Handling both cases when there is 1 or multiple subplots
                if num_clusters > 1:
                    ax[row_idx, col_idx].imshow(cluster_mask, cmap='gray')
                    ax[row_idx, col_idx].set_title(f'Cluster {label}')
                else:
                    ax.imshow(cluster_mask, cmap='gray')
                    ax.set_title(f'Cluster {label}')

            plt.suptitle('K-Means Clustering' + f' (k={num_clusters})' + f' - {os.path.basename(model_path)}')
            plt.tight_layout()
            plt.show()

        if select_smaller_cluster:
            cluster_counts = [np.sum(labels == cluster_label) for cluster_label in unique_labels]
            smaller_cluster_label = unique_labels[np.argmin(cluster_counts)]
            cluster_mask = (labels == smaller_cluster_label).reshape(img.shape).astype(np.uint8)
        else:
            cluster_mask = (labels == selected_cluster).reshape(img.shape).astype(np.uint8)
        return cluster_mask

    def create_phase_mask(self, img, show_images=False):
        thr_img = self._load_thresholded_image(self.img_path)
        model_name, model_path = self._get_model_name_and_path(0)
        clustered_binary_image = self._get_best_mask(img, thr_img, model_name)
        hole_filled_mask = fill_holes(clustered_binary_image)
        filtered_mask = filter_islands_by_size(hole_filled_mask)

        if show_images:
            # Create a plot with subplots for each step
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(img, cmap='gray')
            axs[0].set_title('Original Image')

            axs[1].imshow(thr_img, cmap='gray')
            axs[1].set_title('Thresholded Image')

            axs[2].imshow(filtered_mask, cmap='gray')
            axs[2].set_title('Filtered Mask')

            fig.suptitle('KMeans Image Segmentation')

            plt.show()

        return filtered_mask

    def create_mng_mask(self, img, show_images=False):
        clustered_binary_image = self._perform_kmeans_clustering(img, 1, None, show_images, True)

        hole_filled_mask = fill_holes(clustered_binary_image)
        filtered_mask = filter_islands_by_size(hole_filled_mask)

        return filtered_mask

    def create_dna_mask(self, img, show_images=False):
        clustered_binary_image = self._perform_kmeans_clustering(img, 2, None, show_images, True)

        hole_filled_mask = fill_holes(clustered_binary_image)
        filtered_mask = filter_islands_by_size(hole_filled_mask)

        return filtered_mask

    def _load_thresholded_image(self, image_path):
        # Replace the file extension with '_thr.tif' to get the thresholded file's path.
        thresholded_image_path = image_path.replace('.tif', '_thr.tif')
        thr_img = plt.imread(thresholded_image_path)  # Assuming plt is matplotlib.pyplot
        return thr_img

    def _compute_overlap(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        overlap = np.sum(intersection) / np.sum(union)
        return overlap

    def _get_best_mask(self, img, thr_img, classifier_filename):
        num_clusters = self._get_num_clusters_from_filename(classifier_filename)
        max_overlap = 0
        best_mask = None

        for cluster_index in range(1, num_clusters + 1):  # +1 because range is exclusive at the end
            clustered_binary_image = self._perform_kmeans_clustering(img, 0, cluster_index, False)
            overlap = self._compute_overlap(clustered_binary_image, thr_img)
            if overlap > max_overlap:
                max_overlap = overlap
                best_mask = clustered_binary_image
        return best_mask

    def _get_num_clusters_from_filename(self, classifier_filename):
        """
        Given a kmeans classifier filename, extracts and returns the number of clusters.
        Assumes filename format: 'Tb927.3.4290_channel_0_clusters_X.joblib' where X is the number of clusters.
        """
        classifier_filename = classifier_filename.split(".")[2]
        return int(classifier_filename.split("_")[-1])


@ray.remote
def process_tif_file(tif_file_name, directory):
    field_name = os.path.splitext(tif_file_name)[0]
    gene_name, _, terminus, field_index = field_name.split('_')
    field_directory = os.path.join(directory, field_name)
    if not os.path.exists(field_directory):
        os.makedirs(field_directory)
    field = Field(index=int(field_index), terminus=terminus, gene_name=gene_name, base_path=field_directory,
                  img_path=os.path.join(directory, tif_file_name),
                  mask_path=os.path.join(directory, field_name + "_thr.tif"))
    return field


@dataclass
class TrypTagDataset:
    directory: str
    genes_of_interest: Optional[List[str]] = None
    fields: List[Field] = None

    def __post_init__(self):
        self.fields = []
        tif_files = [file.name for file in os.scandir(self.directory) if
                     file.name.endswith('.tif') and "thr" not in file.name and
                     any(gene in file.name for gene in (self.genes_of_interest or []))]

        futures = [process_tif_file.remote(tif_file, self.directory) for tif_file in tif_files]

        pbar = tqdm(total=len(futures), desc="Processing TIFF files")
        while len(futures):
            done, futures = ray.wait(futures)
            results = ray.get(done)
            self.fields.extend(results)
            pbar.update(len(done))
        pbar.close()

    def total_number_of_cells(self):
        return sum([len(field.cells) for field in self.fields])


def stack_images_and_train_kmeans(directory, channel, n_clusters, genes_of_interest=None, show_images=False):
    files = list(set([file for file in os.listdir(directory) if file.endswith('.tif') and 'thr.tif' not in file
                      and any(gene in file for gene in (genes_of_interest or []))]))

    # dont set k too high, you will run out of memory
    files = random.sample(files, k=max(len(files), min(30, len(files))))
    image_pixels = 2560 * 2160
    print(f"Training KMeans model on {len(files) * image_pixels} datapoints")

    stacked_images = np.zeros((image_pixels * len(files), 1))

    for i, file_name in enumerate(files):
        img = tiff.imread(os.path.join(directory, file_name))
        img_channel = img[channel, :, :].reshape(-1, 1)
        stacked_images[image_pixels * i:image_pixels * (i + 1), :] = img_channel

    flattened_images = stacked_images.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_images)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)

    gene_id = directory.split("\\")[-1]
    model_path = os.path.join("kmeans_models", f'{gene_id}_channel_{channel}_clusters_{n_clusters}.joblib')
    dump(kmeans, model_path)

    print(f"KMeans model trained and saved to {model_path}")

    if show_images:
        img_chunk = flattened_images[:image_pixels].reshape(2160, 2560)
        labels_chunk = labels[:image_pixels].reshape(2160, 2560)

        num_clusters = len(unique_labels)
        num_cols = 3  # You can adjust this
        num_rows = max(num_clusters // num_cols, 1)
        if num_clusters % num_cols:
            num_rows += 1  # Add an extra row if the clusters cannot be arranged evenly

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 3))
        if num_clusters == 1:
            ax = np.array([ax])  # Convert ax to a 1D numpy array for consistent indexing

        for i, label in enumerate(unique_labels):
            cluster_mask = (labels_chunk == label)

            row_idx = i // num_cols
            col_idx = i % num_cols

            # Handling both cases when there is 1 or multiple subplots
            if num_rows > 1:
                ax[row_idx, col_idx].imshow(img_chunk * cluster_mask, cmap='gray')
                ax[row_idx, col_idx].set_title(f'Cluster {label}')
            else:
                ax[col_idx].imshow(img_chunk * cluster_mask, cmap='gray')
                ax[col_idx].set_title(f'Cluster {label}')

        plt.suptitle('K-Means Clustering' + f' (k={num_clusters})' + f' - {os.path.basename(model_path)}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    base_directory = r'E:\PycharmProjects\BA\dataset\raw'

    # train 1 KMeans models for each channel and each gene
    # phase, channel 0, clusters 4
    # mng, channel 1, clusters 2
    # dna, channel 2, clusters 2

    selected_genes = SELECTED_GENES

    # Define channels and their respective cluster settings
    channels_and_clusters = {
        0: 4,  # phase
        1: 2,  # mng
        2: 2  # dna
    }


    def train_and_save_all(clustering=True):
        # Train a KMeans model for each gene and each channel
        for gene in selected_genes.keys():
            # Update directory path for the current gene
            gene_directory = os.path.join(base_directory, gene)

            if clustering:
                for channel, n_clusters in channels_and_clusters.items():
                    print(f"Training KMeans for gene {gene}, channel {channel} with {n_clusters} clusters.")
                    stack_images_and_train_kmeans(gene_directory, channel, n_clusters, gene, False)

            TrypTagDataset(directory=gene_directory, genes_of_interest=[gene])


    train_and_save_all(False)

# for gene in selected_genes.keys():
#     gene_directory = os.path.join(base_directory, gene)
#     TrypTagDataset(directory=gene_directory, genes_of_interest=[gene])
