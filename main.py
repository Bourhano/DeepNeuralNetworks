import numpy as np
import gzip
from scipy.io import loadmat
import matplotlib.pyplot as plt


def read_mnist_images(path='./data/MNIST/', train=True,
                      binary=False, thresh=0.5):
    img_file_name = '-images-idx3-ubyte.gz'
    file_name = 'train' + img_file_name if train else 't10k' + img_file_name
    num_images = 60_000 if train else 10_000
    image_size = 28

    full_path = path + file_name
    f = gzip.open(full_path, 'r')

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size)
    if binary:
        data = (data > thresh).astype(int)

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
        label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0]
        labels.append(label)

    return np.array(labels)


def read_mnist(binary=False):
    train_images = read_mnist_images(train=True, binary=binary)
    train_labels = read_mnist_labels(train=True)
    test_images = read_mnist_images(train=False, binary=binary)
    test_labels = read_mnist_labels(train=False)

    return train_images, test_images, train_labels, test_labels


def read_alpha_digit(digit, path='./data/BinaryAlphaDigits/'):
    """
    Returns images corresponding to digit as a stack of 1-D vectors

    :param digit (character) the desired digit
    :param path (string) directory containing the data file
    :return: ndarray of shape 39x320 corresponding to
    the images of :param digit
    """
    file_name = 'binaryalphadigs.mat'
    full_path = path + file_name
    data = loadmat(full_path)

    images = data['dat']
    counts = data['classcounts'][0]
    all_labels = data['classlabels'][0]
    all_labels = np.hstack(all_labels)
    idx = np.argmax(np.array(all_labels == digit))

    digit_counts = np.squeeze(counts[idx])
    digit_imgs = np.hstack(images[idx]).reshape(digit_counts, -1)

    return digit_imgs


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def logit(y):
    return np.log(y / (1 - y))


class RBM:

    def __init__(self, W=None, a=None, b=None, p=None, q=None):
        self.W = W  # projection weights
        self.a = a  # input bias
        self.b = b  # output bias
        self.p = p  # input dimension
        self.q = q  # output dimension

    def forward(self, X):
        projection = np.dot(X, self.W) + self.b
        return sigmoid(-projection)

    def backward(self, Y):
        projection = np.dot(self.W, Y.T).T + self.a
        return sigmoid(-projection)

    def step(self, batch, learning_rate=0.01):
        n_batch = len(batch)

        x0 = batch
        p_yx0 = input_output_RBM(self, x0)

        y0 = np.random.rand(n_batch, self.q) < p_yx0
        p_xy0 = output_input_RBM(self, y0)

        x1 = np.random.rand(n_batch, self.p) < p_xy0
        p_yx1 = input_output_RBM(self, x1)

        grad_W = np.dot(x0.T, p_yx0) - np.dot(x1.T, p_yx1)
        grad_a = np.sum(x0 - x1, axis=0)
        grad_b = np.sum(p_yx0 - p_yx1, axis=0)

        # gradient ascent
        self.W = self.W + learning_rate * grad_W / n_batch
        self.a = self.a + learning_rate * grad_a / n_batch
        self.b = self.b + learning_rate * grad_b / n_batch

        return self


def init_RBM(p=16*20, q=128, sigma=0.01):
    input_dim = p
    hidden_dim = q

    W = np.random.randn(input_dim, hidden_dim) * sigma
    a = np.zeros(input_dim)
    b = np.zeros(hidden_dim)

    rbm = RBM(W, a, b, p, q)
    return rbm


def input_output_RBM(rbm, x):
    return rbm.forward(x)


def output_input_RBM(rbm, y):
    return rbm.backward(y)


def train_RBM(rbm, xtrain, epochs=10, learning_rate=0.01, batch_size=10):
    """
    Train the RBM epochs times.

    :param rbm:
    :param xtrain:
    :param epochs:
    :param learning_rate:
    :param batch_size:
    :return:
    """
    n = len(xtrain)

    y0 = input_output_RBM(rbm, xtrain)
    x0 = output_input_RBM(rbm, y0)
    error = np.mean(np.square(xtrain - x0))
    print(f"Quadratic error (MSE) at epoch  0 is {error:.4} (random init)")

    for epoch in range(epochs):
        np.random.shuffle(xtrain)

        for i in range(0, n, batch_size):
            batch = xtrain[i:i+min(n, batch_size)]
            rbm = rbm.step(batch, learning_rate=learning_rate)

        y = input_output_RBM(rbm, xtrain)
        xgen = output_input_RBM(rbm, y)

        error = np.mean(np.square(xtrain-xgen))
        print(f"Quadratic error (MSE) at epoch {epoch+1:2} is {error:.4}")

    return rbm


def generate_images(rbm, nb_images=10, nb_gibbs_iter=100):
    images = []

    for i in range(nb_images):
        x = (np.random.rand(rbm.p) < 0.5).reshape(1, -1)

        for _ in range(nb_gibbs_iter):
            y = np.random.rand(rbm.q) < input_output_RBM(rbm, x)
            x = np.random.rand(rbm.p) < output_input_RBM(rbm, y)

        images.append(x)

    return np.array(images)


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Team')
    # imgs = read_alpha_digit('B')
    # print(imgs.shape)
    Xtrain, Xtest, ytrain, ytest = read_mnist(binary=True)
    # print(Xtrain.shape, Xtest.shape, ytrain.shape)
    idxs = ytrain == 9
    imgs = Xtrain.reshape(len(Xtrain), -1)[idxs]
    # print(imgs.shape)

    # imgs = read_alpha_digit('0')
    # for im in imgs:
    #     plt.imshow(im.reshape(28, 28))
    #     plt.show()

    rbm = init_RBM(p=28*28, q=64)
    rbm = train_RBM(rbm, imgs, learning_rate=0.3, batch_size=10, epochs=30)
    gen = generate_images(rbm)

    fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(gen[-1].reshape(20, 16))
    # axs[1].imshow(imgs[-1].reshape(20, 16))
    axs[0].imshow(gen[-1].reshape(28, 28))
    axs[1].imshow(imgs[-1].reshape(28, 28))
    # plt.colorbar()
    plt.show()
