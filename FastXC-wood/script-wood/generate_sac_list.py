import os
from tqdm import tqdm

def generate_sac_list(folder_path, output_file):
    with open(output_file, 'w') as output:
        sac_files = [file for file in os.listdir(folder_path) if file.endswith('.sac')]
        for file in tqdm(sac_files, desc="Processing", unit="file"):
            file_path = os.path.join(folder_path, file)
            output.write(file_path + '\n')

if __name__ == "__main__":
    sac_folder = '/home/woodwood/hpc/station_2/ABNH/2018'  # 替换成你的SAC文件夹路径
    output_txt_file = '/home/woodwood/hpc/station_2/station2_sac_file_List/ABNH_2018_sac_file_list.txt'  # 替换成你希望生成的txt文件路径

    generate_sac_list(sac_folder, output_txt_file)
