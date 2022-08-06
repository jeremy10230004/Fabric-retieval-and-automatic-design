import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sample_images(model, latent_dim=1024, label_kind=10, cmap='binary'):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    nc, nr = 10, 10

    noise = np.random.normal(0, 1, (nr * nc, latent_dim))
    gen_label = np.array([[j + 1 for j in range(10)] for i in range(10)]).reshape((100))
    gen_label = tf.one_hot(gen_label, label_kind)
    gen_imgs = model.predict([noise, gen_label])
    gen_imgs = 0.5 * gen_imgs + 0.5  # tanh

    plt.figure(0, (nc * 2, nr * 2))
    i = 0
    for c in range(nc):
        for r in range(nr):
            n = r * nc + c
            plt.subplot(nr, nc, n + 1)
            plt.imshow(gen_imgs[i], cmap=cmap)
            plt.axis("off")
            i += 1
    plt.savefig("Fabric_result.jpg")
    plt.show()


def load_img():
    train_img = np.load("train_img.npy").astype("float32")
    test_img = np.load("test_img.npy").astype("float32")
    x_train = np.concatenate([train_img, test_img])
    x_train = x_train / 255 * 2 + (-1)

    train_label = np.load("train_label.npy")
    test_label = np.load("test_label.npy")
    x_label = np.concatenate([train_label, test_label])

    print(f"dataset.shape = {x_train.shape}")
    print(f"labels.shape = {x_label.shape}")
    return x_train, x_label
