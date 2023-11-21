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

    # dataset_save_folder = "dataset/"
    # dataset = '1Brzu_WIHhyzTDeDGPQXuOWOyFJPkQGp1'
    # dataset_name = 'dataset.zip'
    # if not os.path.exists(dataset_save_folder) : os.makedirs(dataset_save_folder)

    weights_save_folder = "/root/deIdentification-clp/weights"
    weights = '10NhkmOwauDijY7wymt5XMfiWbFX-tnl9'
    weights_name = 'weights.zip'
    if not os.path.exists(weights_save_folder) : os.makedirs(weights_save_folder)
    
    # retinaNet_save_folder = "clp_landmark_detection/weights/"
    # retinaNet = '19r-N9ayoYDkP8HMyyiBo8v6Zz8PxHFZ1'
    # retinaNet_name = 'Resnet50_Final.pth'
    # if not os.path.exists(retinaNet_save_folder) : os.makedirs(retinaNet_save_folder)

    # download_Zip(google_path+dataset, dataset_save_folder+dataset_name)
    download_Zip(google_path+weights, weights_save_folder+weights_name)
    # download_Zip(google_path+retinaNet, retinaNet_save_folder+retinaNet_name)

    # extract_Zip(dataset_save_folder+dataset_name, dataset_save_folder)
    extract_Zip(weights_save_folder+weights_name, weights_save_folder)
