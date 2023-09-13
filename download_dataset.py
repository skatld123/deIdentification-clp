import os
import gdown
import zipfile


def download_Zip(data_path, output, quiet=False):
    if os.path.exists(output):
        print(output + " data already exist!")
        return
    gdown.download(data_path, output=output, quiet=quiet)


def extract_Zip(zip_path, output_path):
    print("Start extract " + zip_path)
    with zipfile.ZipFile(zip_path) as file:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            sub_dirs = [subDir for subDir in os.listdir(
                output_path) if os.path.isdir(os.path.join(output_path, subDir))]
            dir_name = zip_path.split('/')[-1].split('.')[0]
            for sub_dir in sub_dirs:
                if sub_dir == dir_name:
                    print(dir_name + " directory already exist")
                    return
        file.extractall(path=output_path)
        print("Successful extraction " + zip_path + " data")


if __name__ == "__main__":
    google_path = 'https://drive.google.com/uc?id='
    save_folder = "./data/"

    dataset = '17njZGPSMoaz0QDaHM96r0Jz_2RXoDe8a'
    dataset_name = 'dataset_2044_new.zip'

    # lidar_zip_id = '1IYUkvp6hworm33YVQfYPlRhf2G8hirnu'
    # lidar_zip = 'lidar.zip'

    # csv_zip_id = '1HDFKOohrLOkRLGCBqxwe4EFtTZECASO9'
    # csv_zip = 'csv.zip'

    download_Zip(google_path+dataset, save_folder+dataset_name)

    extract_Zip(save_folder+dataset, save_folder)
