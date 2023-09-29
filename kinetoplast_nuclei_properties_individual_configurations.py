import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from matplotlib import patches
from tryptag import TrypTag, CellLine, tryptools
from tryptag.tryptools import cell_morphology_analysis

from constants import SELECTED_GENES

tryptag = TrypTag()

properties = ["minor_axis_length", "major_axis_length", "orientation", "intensity_mean", "intensity_max", "area"]
dir_path = r"_tryptag_cache/"

selected_genes = SELECTED_GENES

# Initialize Ray
ray.init()
configurations = ['KN', 'KKN', 'KKNN']


@ray.remote
def process_file(gene, terminus, field_indices, cell_indices, plotting=False):
    cell_line = CellLine(gene, terminus, "procyclic")

    kinetoplast_data = {config: {prop: [] for prop in properties} for config in configurations}
    nuclei_data = {config: {prop: [] for prop in properties} for config in configurations}

    for field_index in field_indices:
        for cell_index in cell_indices:
            try:
                cell_image = tryptag.open_cell(cell_line, field_index, cell_index)
                morphology_analysis = cell_morphology_analysis(cell_image)
                for obj in morphology_analysis['objects_k'] + morphology_analysis['objects_n']:
                    # Get the configuration for the current object
                    config = morphology_analysis['kn_ordered']
                    for prop in properties:
                        if obj["type"] == "k":
                            kinetoplast_data[config][prop].append(obj[prop])
                        else:
                            nuclei_data[config][prop].append(obj[prop])

                if plotting:
                    plt = tryptools.plot_morphology_analysis(cell_image, morphology_analysis)

                    # Ellipses corresponding to the major and minor axes
                    for obj in morphology_analysis['objects_k'] + morphology_analysis['objects_n']:
                        color = 'red' if obj['type'] == 'k' else 'green'
                        ellipse = patches.Ellipse((obj["centroid"]["y"], obj["centroid"]["x"]),
                                                  obj["major_axis_length"],
                                                  obj["minor_axis_length"],
                                                  angle=-obj["orientation"] * (180 / np.pi) + 90,
                                                  fill=False,
                                                  edgecolor=color,
                                                  label=obj['type'] if 'k' in obj['type'] or 'n' in obj[
                                                      'type'] else None)
                        plt.gca().add_patch(ellipse)

                    # Adding labels, title, and legend
                    plt.xlabel('Position in px')  # replace 'units' with appropriate unit or description
                    plt.ylabel('Position in px')  # replace 'units' with appropriate unit or description
                    plt.title('Cell Morphology Analysis with Ellipses')
                    handles, labels = plt.gca().get_legend_handles_labels()
                    anterior_legend = patches.Patch(color=(1.0, 0.0, 1.0, 0.5), label='Anterior')
                    posterior_legend = patches.Patch(color=(1.0, 1.0, 0.0, 0.5), label='Posterior')

                    handles.extend([anterior_legend, posterior_legend])
                    labels.extend(["Anterior", "Posterior"])
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), title='DNA Type')

                    plt.show()
            except:
                pass

    return {config: (kinetoplast_data[config], nuclei_data[config]) for config in configurations}


def contains_selected_gene(filename):
    for gene, details in selected_genes.items():
        if gene in filename:
            designation = details[0].upper()  # 'n' or 'c' converted to uppercase
            if f"_{designation}_" in filename:  # Ensure the filename has '_N_' or '_C_'
                return True
    return False


all_files = []
for root_dir, subdirs, files in os.walk(dir_path):
    # Skip _zenodo directory
    if "_zenodo" in root_dir:
        continue

    for file in files:
        file_path = os.path.join(root_dir, file)

        if contains_selected_gene(file_path):
            all_files.append(file)

# Filter for only .tif files that do not contain 'thr.tif'
tif_files = list(set([str(file) for file in all_files if file.endswith('.tif') and 'thr.tif' not in file]))

field_indices = range(20)  # Adjust the range according to your field indices
cell_indices = range(800)  # Adjust the range according to your cell indices

# Extract unique combinations
unique_combinations = set()
for file in tif_files:
    gene = file.split("_")[0]
    terminus = file.split("_")[2].lower()
    unique_combinations.add((gene, terminus))

futures = [process_file.remote(gene, terminus, field_indices, cell_indices, False) for gene, terminus in
           unique_combinations]

# Initialize your data structures
kinetoplast_data = {config: {prop: [] for prop in properties} for config in configurations}
nuclei_data = {config: {prop: [] for prop in properties} for config in configurations}

# When you get results back, combine them like so:
for future in futures:
    result = ray.get(future)  # Assuming result is a dictionary {config: (kinetoplast_data, nuclei_data)}
    for config in configurations:
        kinetoplast_data_temp, nuclei_data_temp = result[config]
        for prop in properties:
            kinetoplast_data[config][prop].extend(kinetoplast_data_temp[prop])
            nuclei_data[config][prop].extend(nuclei_data_temp[prop])

# Define custom labels for specific properties
custom_labels = {
    "minor_axis_length": "minor_axis_length in pixels",
    "major_axis_length": "major_axis_length in pixels",
    "orientation": "orientation in radians between the x-axis and the major-axis",
    "intensity_mean": "intensity_mean in arbitrary units",
    "intensity_max": "intensity_max in arbitrary units",
    "area": "area in pixels squared"
}

fixed_limits = {
    "minor_axis_length": 30,
    "major_axis_length": 45,
    "area": 600
}

# Initialize dictionaries to store the number of bins for each configuration and property
num_bins_kinetoplast = {config: {} for config in configurations}
num_bins_nuclei = {config: {} for config in configurations}

# Iterate over each configuration
for config in configurations:
    # Calculate the number of bins for each property within the current configuration
    for prop in properties:
        # Check if the list is not empty before trying to calculate histograms and bins
        if kinetoplast_data[config][prop]:
            _, bins_kinetoplast = np.histogram(kinetoplast_data[config][prop], 'fd')
            num_bins_kinetoplast[config][prop] = len(bins_kinetoplast) - 1  # Adjust this as needed
        else:
            print(f"Warning: No kinetoplast data for {config} - {prop}")
            num_bins_kinetoplast[config][prop] = 1  # or another default value

        # Repeat the same for nuclei data
        if nuclei_data[config][prop]:
            _, bins_nuclei = np.histogram(nuclei_data[config][prop], 'fd')
            num_bins_nuclei[config][prop] = len(bins_nuclei) - 1  # Adjust this as needed
        else:
            print(f"Warning: No nuclei data for {config} - {prop}")
            num_bins_nuclei[config][prop] = 1  # or another default value

font_size = 12  # Adjust as needed
plt.rc('font', size=font_size)  # controls default text sizes
plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)  # legend fontsize

config_names = {"KN": "1K1N", "KKN": "2K1N", "KKNN": "2K2N"}  # Mapping original config names to new names

for config in configurations:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for each configuration plot
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)  # Adjust as needed

    for prop_index, prop in enumerate(properties):
        row = prop_index // 3  # Calculate row index for subplot
        col = prop_index % 3  # Calculate column index for subplot
        ax = axs[row, col]

        kin_data = kinetoplast_data[config][prop]
        nuc_data = nuclei_data[config][prop]

        # Set upper bound if required
        upper_bound = fixed_limits.get(prop, None)

        if upper_bound:
            ax.hist(kin_data, bins=num_bins_kinetoplast[config][prop], alpha=0.5, label='Kinetoplasts', color='r',
                    range=(0, upper_bound))
            ax.hist(nuc_data, bins=num_bins_nuclei[config][prop], alpha=0.5, label='Nuclei', color='b',
                    range=(0, upper_bound))
        else:
            ax.hist(kin_data, bins=num_bins_kinetoplast[config][prop], alpha=0.5, label='Kinetoplasts', color='r')
            ax.hist(nuc_data, bins=num_bins_nuclei[config][prop], alpha=0.5, label='Nuclei', color='b')

        ax.set_xlim([0, upper_bound])
        ax.set_title(f"{config_names[config]}: Distribution of {prop}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        ax.set_xlabel(custom_labels.get(prop, prop))
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def print_stats(data, name):
    # Creating an empty list to hold the rows of the table
    table_rows = []

    # Iterating over configurations and properties, calculating statistics and appending to table_rows
    for config in data.keys():
        for prop in data[config].keys():
            num_samples = len(data[config][prop])
            mean = np.mean(data[config][prop]) if num_samples > 0 else np.nan
            std_dev = np.std(data[config][prop]) if num_samples > 0 else np.nan

            # Appending a dictionary with statistics as a new row to the table_rows list
            table_rows.append(
                {'Configuration': config, 'Property': prop, 'Number of Samples': num_samples, 'Mean': mean,
                 'Standard Deviation': std_dev})

    # Creating a DataFrame from the list of rows
    df = pd.DataFrame(table_rows)

    # Printing the table with a title
    print(f"Statistics for {name}:\n")
    print(df.to_string(index=False))


# Calling the function with your data
print_stats(kinetoplast_data, "Kinetoplasts")
print_stats(nuclei_data, "Nuclei")
