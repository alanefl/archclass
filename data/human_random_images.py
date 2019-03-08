"""Gets 25 random images from the dev directory and saves them into 'human' dir
"""
import random
import os
import shutil

IMAGE_DIR_NAME = 'prepared_arc_dataset'
HUMAN_DIR_NAME = 'human'

def main():
    random.seed(1729)

    file_names = os.listdir(IMAGE_DIR_NAME + '/dev')
    file_names.sort()
    random.shuffle(file_names)
    print(file_names[0:25])
    i = 1
    for file_name in file_names[0:25]:
        shutil.copyfile(
            IMAGE_DIR_NAME + '/dev/' + file_name,
            IMAGE_DIR_NAME + '/' + HUMAN_DIR_NAME + '/' + str(i) + '.jpg'
        )
        i += 1

if __name__ == '__main__':
    main()
