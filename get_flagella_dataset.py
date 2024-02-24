import os
import shutil

from tryptag import TrypTag, CellLine

from constants import SELECTED_GENES, DATASET_DIR


def download_data():
    tryptag = TrypTag()

    for cell_id, cell_info in SELECTED_GENES.items():
        tryptag.fetch_data(CellLine(life_stage=cell_info[1], gene_id=cell_id, terminus=cell_info[0]))


def create_dataset():
    # Define the source directory where you want to search for files
    source_directory = "_tryptag_cache"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # Iterate through all subdirectories in the source directory
    for dir in os.listdir(source_directory):
        if dir == "_zenodo":
            continue

        for file in os.listdir(os.path.join(source_directory, dir)):
            name = file.split("_")[0]
            terminus = file.split("_")[2]

            # If the name is in selected_genes and the terminus matches the expected value
            if name in SELECTED_GENES.keys() and "tif" in file and terminus == SELECTED_GENES[name][0].upper():

                # Create a subdirectory for the gene if it doesn't exist
                gene_directory = os.path.join(DATASET_DIR, name)
                if not os.path.exists(gene_directory):
                    os.makedirs(gene_directory)

                # Create the absolute path of the file by joining the source directory with the file name
                source_file = os.path.join(source_directory, dir, file)

                # Copy the file to the gene's subdirectory within the destination directory
                destination_file = os.path.join(gene_directory, file)
                print("Copying file: " + source_file + " to " + destination_file)
                shutil.copy(source_file, destination_file)

    print("File copying process completed.")


if __name__ == "__main__":
    """
    The download takes a long long time. If you dont want to download all you can just download one gene by uncommenting
    the rest of the selected genes in the constants.py
    
    Create dataset will create a folder /dataset and copy the tiff files into it, that will later be used by
    compute_segmentation_masks_and_dataset.py
    """
    download_data()
    create_dataset()
