import os
import shutil
import random
from os import path, walk

DATA_DIR = 'data'
FLOWER_DIR = path.join(DATA_DIR, 'flowers')


def main():
    for _dir in os.listdir(FLOWER_DIR):
        _path = path.join(FLOWER_DIR, _dir)
        if path.isdir(_path):
            train_path = path.join(DATA_DIR, 'flowers_train', _dir)
            test_path = path.join(DATA_DIR, 'flowers_test', _dir)
            if not path.lexists(train_path):
                print('creating path:', train_path)
                os.makedirs(train_path)

            if not path.lexists(test_path):
                print('creating path:', test_path)
                os.makedirs(test_path)

            for root, dirs, files in walk(_path):
                for f in files:
                    copy_dir = 'flowers_train' if random.random() < 0.9 else 'flowers_test'
                    shutil.copyfile(path.join(root, f), path.join(DATA_DIR, copy_dir, _dir, f))


if __name__ == "__main__":
    main()
