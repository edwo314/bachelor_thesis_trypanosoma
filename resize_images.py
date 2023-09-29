from PIL import Image
import os

def resize_images(input_dir, output_dir, new_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img = img.resize(new_size, resample=Image.BOX)
                    img.save(output_path)
                    print(f"Resized {filename} to {new_size} successfully.")
            except Exception as e:
                print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    input_directory = "phase"  # Replace with your input directory containing PNG files
    output_directory = "phase_resized"  # Replace with your output directory for resized PNG files
    new_size = (100, 100)

    resize_images(input_directory, output_directory, new_size)
