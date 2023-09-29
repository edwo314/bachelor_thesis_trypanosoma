# Bachelor Thesis: Trypanosoma Code Repository

This repository contains the code used for generating plots and statistics in my Bachelor's thesis related to *Trypanosoma*. The code is provided "AS IS", without warranty of completeness or functionality, as it was created for specific, non-general purposes.

## Disclaimer
- The code may not be user-friendly, and it was not intended for use by the general public.
- Users should be cautious and aware that some files need to be run in a specific order due to dependencies on the output of other files.

## Usage

### Initial Setup
1. Clone this repository.
2. Install the required dependencies listed in `requirements.txt` using a virtual environment (recommended). Additional installation instructions for `tryptag`, `umap`, and `intel` extension for `scikit-learn` are included within `requirements.txt`.

### Dataset Preparation
1. Run `get_flagella_dataset.py` to download the necessary dataset.
2. Execute `compute_segmentation_masks_and_dataset.py` to create k-means models for segmentation and generate segmentation masks for the entire dataset.

### Important Files to Run in Order
- `resize_images.py`
- `resnet_image_preprocessing.py`
- `resnet_plot.py`

## File Descriptions

- `clustering_metrics.py`: Generates plots for the Elbow Method and Silhouette Scores for a given input image.
- `compute_segmentation_masks_and_dataset.py`: Creates segmentation masks for the whole dataset and saves them to `dataset/raw`. Additionally, it generates and saves all models used for segmentation under `kmeans_models`.
- `constants.py`: Contains a dictionary with all genes used to identify the flagella.
- `gene_selection.py`: Locates genes that localize the paraflagellar rod and ranks them based on percentile scores available on a specified website.
- `get_flagella_dataset.py`: Downloads the dataset using a dictionary of genes.
- `kinetoplast_nuclei_properties_all_configurations.py` and `kinetoplast_nuclei_properties_individual_configurations.py`: These files generate plots showcasing various kinetoplast and nuclei properties such as minor axis length, major axis length, orientation, mean and max DNA intensity, and area.
- `length_histograms.py`: Generates histograms representing the length of the flagella and cells, including corrected measurements.
- `mask_generation.py`: This script was used for data exploration purposes.
- `multivariate_analysis.py`: Creates plots for multivariate analysis.
- `resize_images.py` and `resnet_image_preprocessing.py`: These files resize and preprocess images for the ResNet model and prepare them for display in interactive Plotly charts.
- `resnet_plot.py`: Generates an interactive plot showcasing the principal components of the feature vector of the ResNet50 model.
- `tryptag_copy.py`: A potentially modified copy of `tryptag.py` from the TrypTag GitHub repository.
- `tryptools_copy.py`: A potentially modified copy of `tryptools.py` from the TrypTag GitHub repository.


### Modified Files from TrypTag Repository
This repository includes modified copies of files originally from the [TrypTag GitHub repository](https://github.com/zephyris/tryptag/tree/main).
- `tryptag_copy.py` and `tryptools_copy.py` are altered to support the `kinetoplast_nuclei_properties_all_configurations.py` and `kinetoplast_nuclei_properties_individual_configurations.py` files by adding necessary information for ellipse reconstruction from DNA masks. Replace the original `tryptag.py` and `tryptools.py` in your virtual environment with these copies and then rename them.

## Contribution & Support
Since this code was used specifically for my Bachelor’s thesis, it is not designed for general use or ongoing support. For reference and replication purposes in academic work, you're welcome to use and modify the code while citing appropriate sources.