from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Conv2DTranspose, Embedding, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from tensorflow import keras
import tensorflow as tf
import numpy as np


class CWGANdiv(keras.Model):
    def __init__(self, DataSet, Label, batch_size=128, latent_dim=(16, 16, 8), color_dim=3):
        super(CWGANdiv, self).__init__()
        self.DataSet = DataSet
        self.Label = Label
        self.label_kind = 10
        self.batch_size = batch_size

        self.latent_dim = latent_dim[0] * latent_dim[1] * latent_dim[2]  # 特徵維度
        self.color_dim = color_dim
        self.image_size = (128, 128, self.color_dim)

        self.dropout_rate = 0.3
        self.clip = 0.01
        self.d_TrainTimes = 5
        self.lambda_ = 2  # gp可換10 div可換2
        self.p = 6

        self.d = self.discriminator()
        # self.d = keras.models.load_model("discriminator_PATH")
        self.d.summary()

        self.g = self.load_generator()
        # self.g = keras.models.load_model("generator_PATH")
        self.g.summary()

    @staticmethod
    def w_distance(fake, real=None):
        # the distance of two data distributions
        if real is None:
            return -tf.reduce_mean(fake)
        else:
            return -(tf.reduce_mean(fake) - tf.reduce_mean(real))

    def discriminator(self):
        input_layer = Input(self.image_size)

        input_label = Input(self.label_kind, )
        label_emb = Embedding(self.label_kind, 64)(input_label)
        label_f = Flatten()(label_emb)
        label_dense = Dense(128 * 128 * 3, activation=LeakyReLU(0.2))(label_f)
        label_r = Reshape((128, 128, 3))(label_dense)

        con = concatenate([input_layer, label_r])

        en = Conv2D(32, (4, 4), strides=2, activation=LeakyReLU(0.2))(con)
        en = Dropout(self.dropout_rate)(en)
        en = Conv2D(64, (4, 4), strides=2, activation=LeakyReLU(0.2))(en)
        en = Dropout(self.dropout_rate)(en)
        en = Conv2D(64, (4, 4), strides=2, activation=LeakyReLU(0.2))(en)
        en = Dropout(self.dropout_rate)(en)

        f = Flatten()(en)
        ans = Dense(1)(f)
        return Model([input_layer, input_label], ans, name="discriminator")

    def generator(self):
        input_layer = Input(self.latent_dim, )

        input_label = Input(self.label_kind, )
        label_emb = Embedding(self.label_kind, 64)(input_label)
        label_f = Flatten()(label_emb)
        label_dense = Dense(16 * 16 * 8, activation=LeakyReLU(0.2))(label_f)

        con = concatenate([input_layer, label_dense])
        d = Dense(16 * 16 * 8, activation="relu")(con)  # 11111111111111111
        d = Dense(16 * 16 * 16, activation="relu")(d)
        r = Reshape((16, 16, 16))(d)
        d = Conv2DTranspose(64, (4, 4), strides=2, activation="relu", padding='same')(r)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64, (4, 4), strides=2, activation="relu", padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(32, (4, 4), strides=2, activation="relu", padding='same')(d)
        d = BatchNormalization()(d)
        ans = Conv2DTranspose(self.color_dim, (4, 4), activation='tanh', padding='same')(d)
        return Model([input_layer, input_label], ans, name="generator")

    def load_generator(self):
        input_layer = Input(self.latent_dim, )

        input_label = Input(self.label_kind, )
        label_emb = Embedding(self.label_kind, 64)(input_label)
        label_f = Flatten()(label_emb)
        label_dense = Dense(16 * 16 * 8, activation=LeakyReLU(0.2))(label_f)

        con = concatenate([input_layer, label_dense])
        d = Dense(16 * 16 * 8, activation="relu", name="Dense_g1")(con)

        con = concatenate([input_layer, label_dense])
        c = Dense(2048, activation="relu", name="Dense_g2")(con)
        # -------------------
        decoder = keras.models.load_model("decoder_PATH")
        decoder.summary()

        de = decoder.layers[1](c)
        de = decoder.layers[2](de)
        de = decoder.layers[3](de)
        de = decoder.layers[4](de)
        de = decoder.layers[5](de)
        ans = decoder.layers[6](de)

        return Model([input_layer, input_label], ans, name="generator")

    def generate_img(self, label, training=False):
        # gen_label = tf.one_hot(np.random.randint(0, self.label_kind, self.batch_size), self.label_kind)
        return self.g.call([tf.random.normal((self.batch_size, self.latent_dim)), label], training=training)

    def gradient_penalty(self, real_img, fake_img, real_label):
        e1 = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1, dtype='float32')
        noise_img = e1 * real_img + (1. - e1) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            tape.watch(real_label)
            o = self.d([noise_img, real_label])
        g = tape.gradient(o, noise_img)  # image gradients
        gp = tf.pow(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]), self.p)
        return tf.reduce_mean(gp)

    def TrainG(self, sample_label):
        with tf.GradientTape() as tape:
            gen_data = self.generate_img(sample_label, training=True)
            gd_ans = self.d.call([gen_data, sample_label], training=False)
            g_loss = self.w_distance(gd_ans) * (-1)  # 最小化w距離
        g_grads = tape.gradient(g_loss, self.g.trainable_variables)
        self.optimizer.apply_gradients(zip(g_grads, self.g.trainable_variables))
        return g_loss

    def TrainD(self, sample_data, sample_label):
        with tf.GradientTape() as tape:
            gen_data = self.generate_img(sample_label, training=False)
            d_fake = self.d.call([gen_data, sample_label], training=True)
            d_real = self.d.call([sample_data, sample_label], training=True)
            w_loss = self.w_distance(d_fake, d_real)  # 最大化w距離
            gp = self.gradient_penalty(sample_data, gen_data, sample_label)
            d_loss = w_loss + self.lambda_ * gp
        d_grads = tape.gradient(d_loss, self.d.trainable_variables)
        self.optimizer.apply_gradients(zip(d_grads, self.d.trainable_variables))
        return d_loss

    def StepOfTrain(self, now_epoch):
        # train Discriminator
        for times in range(self.d_TrainTimes):
            idx = np.random.randint(0, self.DataSet.shape[0], self.batch_size)
            sample_data = tf.gather(self.DataSet, idx)
            sample_label = tf.gather(self.Label, idx)
            sample_label = tf.one_hot(sample_label, self.label_kind)
            d_loss = self.TrainD(sample_data, sample_label)
            # clip Discriminator
        # train Generator
        g_loss = self.TrainG(sample_label)
        print(f"Epoch {now_epoch} G loss : {g_loss.numpy()}, D loss : {d_loss.numpy()}")

    def Train(self, steps, batch_size=32):
        self.batch_size = batch_size
        for e in range(steps):
            self.StepOfTrain(e)

