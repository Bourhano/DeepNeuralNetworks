import numpy as np
import gzip
from scipy.io import loadmat


def read_mnist_images(path='./data/MNIST/', train=True):
    img_file_name = '-images-idx3-ubyte.gz'
    file_name = 'train' + img_file_name if train else 't10k' + img_file_name
    num_images = 60_000 if train else 10_000
    image_size = 28

    full_path = path + file_name
    f = gzip.open(full_path, 'r')

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    return data


def read_mnist_labels(path='./data/MNIST/', train=True):
    labels_file_name = '-labels-idx1-ubyte.gz'
    file_name = 'train' + labels_file_name if train else 't10k' + labels_file_name
    num_labels = 60_000 if train else 10_000

    full_path = path + file_name
    f = gzip.open(full_path, 'r')

    f.read(8)
    labels = []
    for i in range(num_labels):
        buf = f.read(1)
        label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels.append(label)

    return np.array(labels)


def read_mnist():
    train_images = read_mnist_images(train=True)
    train_labels = read_mnist_labels(train=True)
    test_images = read_mnist_images(train=False)
    test_labels = read_mnist_labels(train=False)

    return train_images, test_images, train_labels, test_labels


def read_alpha_digit(digit, path='./data/BinaryAlphaDigits/'):
    """

    :param digit: the desired digit
    :param path: directory containing the data file
    :return: images corresponding to digit
                in a ndarray of shape 39x320
    """
    file_name = 'binaryalphadigs.mat'
    full_path = path + file_name
    data = loadmat(full_path)

    images = data['dat']
    counts = data['classcounts'][0]
    all_labels = data['classlabels'][0]
    all_labels = np.array([l[0] for l in all_labels])
    idx = np.argmax(np.array(all_labels == digit))

    digit_counts = np.squeeze(counts[idx])
    digit_imgs = np.hstack(images[idx]).reshape(digit_counts, -1)

    return digit_imgs


class RBM:

    def __init__(self):
        self.W = None  # projection weights
        self.a = None  # input bias
        self.b = None  # output bias


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Team')
    # imgs = read_alpha_digit('A')
    # print(imgs.shape)
    Xtrain, Xtest, ytrain, ytest = read_mnist()
    print(Xtrain.shape, Xtest.shape)
