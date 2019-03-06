"""Download dataset, Split Architecture Style dataset into train/dev/test,
and resize images to 512 x 512.


Resizing:
    Original images vary in sizes.


"""
import random
import requests
import os
import zipfile

from PIL import Image
from tqdm import tqdm

SIZE = 512
ZIP_FILE_NAME = "_arc_dataset.zip"
DATA_DIR = "_arc_dataset"
RESIZED_OUTPUT_DIR = "prepared_arc_dataset"
OUTPUT_DIR = "prepared_arc_dataset_not_resized"
GDRIVE_FILE_ID = '1fJfclq0ULmSX_E6g6qeO2Ff0DtcUW1eh'


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def resize_and_save(filename, output_dir, should_resize=False, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`
    """
    image = Image.open(filename)

    # Use bilinear interpolation instead of the default "nearest neighbor" method
    if (should_resize):
        image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


def rename_files():
    """Rename image files to the following convention:
          [architecture style]-[id].jpg

    NOTE: this function is not needed since the data on GDrive is already renamed.
    """
    arch_style_dirs = [x[0] for x in os.walk(DATA_DIR) if x[0] != DATA_DIR]
    for arch_style_dir in arch_style_dirs:
        _, arch_style = arch_style_dir.split("/")
        ctr = 0
        for file_name in os.listdir(arch_style_dir):
            new_file_name = "%s-0%d.jpg" % (arch_style, ctr)
            ctr += 1
            os.rename(
                arch_style_dir + "/" + file_name,
                arch_style_dir + "/" + new_file_name
            )


def fetch_dataset():
    """Fetches prepared Architecture Style dataset and saves it locally as a zip file.
    """
    download_file_from_google_drive(GDRIVE_FILE_ID, ZIP_FILE_NAME)


def unzip_original_dataset():
    """Unzips downloaded dataset.
    """
    print("Unzipping file...")
    zip_ref = zipfile.ZipFile(ZIP_FILE_NAME, 'r')
    zip_ref.extractall(".")
    zip_ref.close()
    print("Unzipped file.")


def train_dev_test_split(train=.7, dev=.2, test=.1):
    """Creates a random train/dev/test split and installs it
    in the OUTPUT_DIR directory.

    The percentages of train, dev, and test are taken PER CLASS,
    in order to preserve the distributions of each class in each =
    of the sets.

    :param train: Percentage to use for training set
    :param dev:   Percentage to use for dev set
    :param test:  Percentage to use for test set
    :return:
    """
    assert((train + dev + test) - 1 < 1E-5)

    arch_style_dirs = [x[0] for x in os.walk(DATA_DIR) if x[0] != DATA_DIR]

    file_names_per_label = {}
    for arch_style_dir in arch_style_dirs:
        _, arch_style = arch_style_dir.split("/")
        file_names_per_label[arch_style] = []
        for file_name in os.listdir(arch_style_dir):
            file_names_per_label[arch_style].append(
                os.path.join(DATA_DIR, file_name.split('-')[0], file_name)
            )

    for label in file_names_per_label:
        random.shuffle(file_names_per_label[label])

    train_file_names = []
    dev_file_names = []
    test_file_names = []

    for label in file_names_per_label:
        random.shuffle(file_names_per_label[label])
        train_split = int(train * len(file_names_per_label[label]))
        dev_split = int((train + dev) * len(file_names_per_label[label]))

        train_file_names += file_names_per_label[label][:train_split]
        dev_file_names += file_names_per_label[label][train_split: dev_split]
        test_file_names += file_names_per_label[label][dev_split:]

    train_file_names.sort()
    dev_file_names.sort()
    test_file_names.sort()

    file_names = {
        'train': train_file_names,
        'dev': dev_file_names,
        'test': test_file_names
    }

    # These datasets must be distinct.
    for file_name in file_names['train']:
        assert(file_name not in file_names['dev'] and file_name not in file_names['test'])

    for file_name in file_names['dev']:
        assert(file_name not in file_names['train'] and file_name not in file_names['test'])

    for file_name in file_names['test']:
        assert(file_name not in file_names['dev'] and file_name not in file_names['train'])

    for should_resize, output_dir in [(True, RESIZED_OUTPUT_DIR), (False, OUTPUT_DIR)]:

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            print("Warning: output dir {} already exists".format(output_dir))

        # Split into train, dev, and test
        for split in ['train', 'dev', 'test']:
            output_dir_split = os.path.join(output_dir, '{}'.format(split))
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_dir_split))

            print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
            for file_name in tqdm(file_names[split]):
                resize_and_save(file_name, output_dir_split, should_resize=should_resize, size=SIZE)


def main():
    random.seed(1729)

    print("Fetching dataset...")
    fetch_dataset()
    print("Got dataset.")
    unzip_original_dataset()

    # print("Renaming filenames and doing train/dev/test split...")
    # rename_files()

    # Split into dev and test
    train_dev_test_split(OUTPUT_DIR)
    print("Done building dataset.")


if __name__ == '__main__':
    main()
