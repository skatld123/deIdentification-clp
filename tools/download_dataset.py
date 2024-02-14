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

    weights_save_folder = "/root/deIdentification-clp/weights/"
    weights = '1gG4usDypUkZxaaBlf1mIeoCLEquQ1asF'
    weights_name = 'weight_final.zip'
    if not os.path.exists(weights_save_folder) : os.makedirs(weights_save_folder)

    download_Zip(google_path+weights, weights_save_folder+weights_name)

    extract_Zip(weights_save_folder+weights_name, weights_save_folder)
