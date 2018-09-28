import tarfile
from fastai import *
from fastai.vision import *

DATA_PATH = Path('..')/'data'
MNIST_PATH = DATA_PATH / 'mnist_sample'

def get_mnist():
    if not MNIST_PATH.exists():
        tarfile.open(MNIST_PATH.with_suffix('.tgz'), 'r:gz').extractall(DATA_PATH)
    return image_data_from_folder(MNIST_PATH)

