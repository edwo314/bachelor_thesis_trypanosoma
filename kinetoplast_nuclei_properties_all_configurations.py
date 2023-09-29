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


@ray.remote
def process_file(gene, terminus, field_indices, cell_indices, plotting=False):
    cell_line = CellLine(gene, terminus, "procyclic")
    kinetoplast_data = {prop: [] for prop in properties}
    nuclei_data = {prop: [] for prop in properties}
    for field_index in field_indices:
        for cell_index in cell_indices:
            try:
                cell_image = tryptag.open_cell(cell_line, field_index, cell_index)
                # tryptools.cell_kn_analysis(cell_image)
                morphology_analysis = cell_morphology_analysis(cell_image)
                for obj in morphology_analysis['objects_k'] + morphology_analysis['objects_n']:
                    # Collect data for histograms
                    for prop in properties:
                        if obj["type"] == "k":
                            kinetoplast_data[prop].append(obj[prop])
                        else:
                            nuclei_data[prop].append(obj[prop])
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

    return kinetoplast_data, nuclei_data


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

# Create a list to store the results
results = []

# Iterate over the futures and retrieve the results
for future in futures:
    result = ray.get(future)
    results.append(result)

# Combine the results into a single dictionary
kinetoplast_data = {prop: [] for prop in properties}
nuclei_data = {prop: [] for prop in properties}

for result in results:
    kinetoplast_data_temp, nuclei_data_temp = result
    for prop in properties:
        kinetoplast_data[prop].extend(kinetoplast_data_temp[prop])
        nuclei_data[prop].extend(nuclei_data_temp[prop])

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution of Cell Properties", fontsize=16)

# Determine appropriate number of bins separately for kinetoplast and nuclei
num_bins_kinetoplast = {}
num_bins_nuclei = {}
for prop in properties:
    _, bins_kinetoplast = np.histogram(kinetoplast_data[prop], 'fd')  # Using Freedman-Diaconis rule for kinetoplasts
    _, bins_nuclei = np.histogram(nuclei_data[prop], 'fd')
    num_bins_kinetoplast[prop] = len(bins_kinetoplast)
    num_bins_nuclei[prop] = len(bins_nuclei)

print(len(kinetoplast_data['minor_axis_length']))

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
# Add histograms to subplots with appropriate number of bins
for i, prop in enumerate(properties):
    # Kinetoplast data
    kin_data = kinetoplast_data[prop]

    # Nuclei data
    nuc_data = nuclei_data[prop]

    row = i // 3
    col = i % 3

    upper_bound = None
    if prop == "minor_axis_length":
        upper_bound = fixed_limits[prop]
    elif prop == "major_axis_length":
        upper_bound = fixed_limits[prop]
    elif prop == "area":
        upper_bound = fixed_limits[prop]

    if upper_bound:
        axs[row, col].hist(kin_data, bins=num_bins_kinetoplast[prop], alpha=0.5, label='Kinetoplasts',
                           color='r', range=(0, upper_bound))
        axs[row, col].hist(nuc_data, bins=num_bins_nuclei[prop], alpha=0.5, label='Nuclei',
                           color='b', range=(0, upper_bound))
    else:
        axs[row, col].hist(kin_data, bins=num_bins_kinetoplast[prop], alpha=0.5, label='Kinetoplasts',
                           color='r')
        axs[row, col].hist(nuc_data, bins=num_bins_nuclei[prop], alpha=0.5, label='Nuclei',
                           color='b')

    axs[row, col].set_xlim([0, upper_bound])
    axs[row, col].set_title(f"Distribution of {prop}")
    axs[row, col].grid(axis='y', linestyle='--', alpha=0.7)
    axs[row, col].legend(loc='upper right')
    axs[row, col].set_xlabel(custom_labels.get(prop, prop))
    axs[row, col].set_ylabel("Frequency")
    axs[row, col].legend(loc='upper right')

plt.suptitle("Distribution of kinetoplast and nuclei properties", fontsize=16)
plt.tight_layout()
plt.show()


def print_stats(data, name):
    # Creating an empty list to hold the rows of the table
    table_rows = []

    # Iterating over properties, calculating statistics and appending to table_rows
    for prop in data.keys():
        num_samples = len(data[prop])
        mean = np.mean(data[prop]) if num_samples > 0 else np.nan
        std_dev = np.std(data[prop]) if num_samples > 0 else np.nan

        # Appending a dictionary with statistics as a new row to the table_rows list
        table_rows.append(
            {'Property': prop, 'Number of Samples': num_samples, 'Mean': mean, 'Standard Deviation': std_dev})

    # Creating a DataFrame from the list of rows
    df = pd.DataFrame(table_rows)

    # Printing the table with a title
    print(f"Statistics for {name}:\n")
    print(df.to_string(index=False))


# Calling the function with your data
print_stats(kinetoplast_data, "Kinetoplasts")
print_stats(nuclei_data, "Nuclei")
