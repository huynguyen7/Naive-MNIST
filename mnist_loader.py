import requests
import numpy as np
import gzip
import os.path
from tqdm import tqdm

train_imgs_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_imgs_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

def load_mnist_data(root='./data/', download=False):
    if download:
        print('Downloading dataset..')
        pbar = tqdm(total=4)
        if not os.path.exists(root):
            os.mkdir(root)  # Create folder.

        train_imgs = requests.get(train_imgs_url, allow_redirects=True).content
        open(f'{root}train-images-idx3-ubyte.gz', 'wb').write(train_imgs)
        pbar.update(1)

        train_labels = requests.get(train_labels_url, allow_redirects=True).content
        open(f'{root}train-labels-idx1-ubyte.gz', 'wb').write(train_labels)
        pbar.update(1)

        test_imgs = requests.get(test_imgs_url, allow_redirects=True).content
        open(f'{root}t10k-images-idx3-ubyte.gz', 'wb').write(test_imgs)
        pbar.update(1)

        test_labels = requests.get(test_labels_url, allow_redirects=True).content
        open(f'{root}t10k-labels-idx1-ubyte.gz', 'wb').write(test_labels)
        pbar.update(1)

        pbar.close()
    else:
        try:
            train_imgs = open(f'{root}train-images-idx3-ubyte.gz', 'rb').read()
            train_labels = open(f'{root}train-labels-idx1-ubyte.gz', 'rb').read()
            test_imgs = open(f'{root}t10k-images-idx3-ubyte.gz', 'rb').read()
            test_labels = open(f'{root}t10k-labels-idx1-ubyte.gz', 'rb').read()
        except:
            print('Data is not downloaded.. Please use flag `download=True`')
            return
    
    # Matrix transformation, using dtype 'int8' since there is only 256 colors for a pixel.
    X_train = np.frombuffer(gzip.decompress(train_imgs), dtype=np.uint8).copy()[0x10:].reshape((-1,28,28))
    Y_train = np.frombuffer(gzip.decompress(train_labels), dtype=np.uint8).copy()[8:]
    X_test = np.frombuffer(gzip.decompress(test_imgs), dtype=np.uint8).copy()[0x10:].reshape((-1,28,28))
    Y_test = np.frombuffer(gzip.decompress(test_labels), dtype=np.uint8).copy()[8:]

    return X_train, Y_train, X_test, Y_test