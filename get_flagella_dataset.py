from tryptag import TrypTag, CellLine
from constants import SELECTED_GENES
import os
import shutil

selected_genes = SELECTED_GENES

def download_data():
    tryptag = TrypTag()

    for cell_id, cell_info in selected_genes.items():
        tryptag.fetch_data(CellLine(life_stage=cell_info[1], gene_id=cell_id, terminus=cell_info[0]))

def create_dataset():
    # Define the source directory where you want to search for files
    source_directory = "_tryptag_cache"

    # Define the destination directory where you want to copy the selected files
    destination_directory = "dataset/raw"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterate through all subdirectories in the source directory
    for dir in os.listdir(source_directory):
        if dir == "_zenodo":
            continue

        for file in os.listdir(os.path.join(source_directory, dir)):
            name = file.split("_")[0]
            terminus = file.split("_")[2]

            # If the name is in selected_genes and the terminus matches the expected value
            if name in selected_genes.keys() and "tif" in file and terminus == selected_genes[name][0].upper():

                # Create a subdirectory for the gene if it doesn't exist
                gene_directory = os.path.join(destination_directory, name)
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
    download_data()
    create_dataset()