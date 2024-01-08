import os
from tqdm import tqdm

def generate_segspec_list(array_dir, output_file):
    # Check if the array directory exists
    if not os.path.exists(array_dir) or not os.path.isdir(array_dir):
        print(f"Error: Directory '{array_dir}' not found.")
        return

    # Create a list to store segspec file paths
    segspec_paths = []

    # Iterate over subdirectories in the array directory
    for subdirectory in tqdm(os.listdir(array_dir), desc='Processing Subdirectories', unit='subdirectory'):
        subdirectory_path = os.path.join(array_dir, subdirectory)

        # Check if the item is a directory
        if os.path.isdir(subdirectory_path):
            # Check if any file in the subdirectory has the .segspec extension
            segspec_file_path = next((f for f in os.listdir(subdirectory_path) if f.endswith('.segspec')), None)
            if segspec_file_path:
                segspec_paths.append(os.path.join(subdirectory_path, segspec_file_path))

    # Open the output file for writing
    with open(output_file, 'w') as file:
        # Write the segspec file paths to the output file
        for segspec_path in tqdm(segspec_paths, desc='Writing to File', unit='file'):
            file.write(segspec_path + '\n')

# Example usage:
array_directory = '/home/woodwood/hpc/station_2/ncf_hinet_AAKH_ABNH/segspec/array2'
output_txt_file = '/home/woodwood/hpc/station_2/z-test/specList_array2.txt'

generate_segspec_list(array_directory, output_txt_file)
