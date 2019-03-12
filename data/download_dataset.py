"""Downloads prepared dataset, split into train, dev, test, and ready for training.
"""
import random
import requests
import zipfile

from tqdm import tqdm

ZIP_FILE_NAME = "prepared_arc_dataset.zip"
DATA_DIR = "_arc_dataset"
RESIZED_OUTPUT_DIR = "prepared_arc_dataset"
OUTPUT_DIR = "prepared_arc_dataset_not_resized"
GDRIVE_FILE_ID = '1xE6WXa_-KW8mjHPkI0Olvb_Js64IVJLn'


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


def main():
    random.seed(1729)

    print("Fetching dataset...")
    fetch_dataset()
    print("Got dataset.")
    unzip_original_dataset()

    print("Done downloading dataset.")


if __name__ == '__main__':
    main()
