import os

import cv2
import ray
from PIL import Image
from tryptag import CellLine, TrypTag

from constants import SELECTED_GENES


@ray.remote
def process_gene(gene, terminus):
    cell_line = CellLine(gene, terminus, "procyclic")

    for field_index in range(20):
        for cell_index in range(800):
            try:
                cell_image = tryptag.open_cell(cell_line, field_index, cell_index)
                phase = (cell_image.phase / 256).astype("uint8")
                mask = cell_image.phase_mask
                mask = (mask > 0).astype("uint8")
                processed_image = phase * mask

                filename = f"{output_dir}/{gene}_{terminus}_{field_index}_{cell_index}.png"
                cv2.imwrite(filename, processed_image)
            except:
                continue


def resize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(input_dir, filename))
            img_resized = img.resize((224, 224))
            img_resized.save(os.path.join(output_dir, filename))


if __name__ == "__main__":
    output_dir = "phase_for_resnet"
    resized_output_dir = "phase_for_resnet_resized"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tryptag = TrypTag()
    selected_genes = SELECTED_GENES

    ray.init()
    futures = [process_gene.remote(gene, selected_genes[gene][0]) for gene in selected_genes.keys()]
    ray.get(futures)
    ray.shutdown()

    resize_images(output_dir, resized_output_dir)
