import Data_Preprocess as pp
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.layers import Reshape, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model


def get_model():
    input_layers1 = Input((64, 64, 3))
    en1 = Conv2D(32, (4, 4), padding='same', strides=2, activation='relu')(input_layers1)
    en1 = Conv2D(16, (4, 4), padding='same', strides=2, activation='relu')(en1)
    en1 = Conv2D(16, (4, 4), padding='same', strides=2, activation='relu')(en1)
    f1 = Flatten()(en1)

    input_layers2 = Input((64, 64, 3))
    en2 = Conv2D(32, (4, 4), padding='same', strides=2, activation='relu')(input_layers2)
    en2 = Conv2D(16, (4, 4), padding='same', strides=2, activation='relu')(en2)
    f2 = Flatten()(en2)

    input_layers3 = Input((64, 64, 3))
    en3 = Conv2D(32, (4, 4), padding='same', strides=2, activation='relu')(input_layers3)
    f3 = Flatten()(en3)

    con = concatenate([f1, f2, f3])
    d = Dense(8 * 8 * 16, activation='relu')(con)
    feature = Dense(512, activation='sigmoid')(d)
    d = Dense(8 * 8 * 16, activation='relu')(feature)
    r = Reshape((8, 8, 16))(d)
    de = Conv2DTranspose(16, (4, 4), padding='same', strides=2, activation='relu')(r)
    de = Conv2DTranspose(16, (4, 4), padding='same', strides=2, activation='relu')(de)
    de = Conv2DTranspose(32, (4, 4), padding='same', strides=2, activation='relu')(de)
    output_layers = Conv2DTranspose(3, (4, 4), padding='same', activation='sigmoid')(de)

    ae = Model([input_layers1, input_layers2, input_layers3], output_layers)
    ae.compile(optimizer='adam', loss='mse')
    ae.summary()

    encoder = Model([input_layers1, input_layers2, input_layers3], feature)
    encoder.compile()
    encoder.summary()
    return ae, encoder


def train(batch_size=64, epochs=3):
    x_data, y_data = pp.load_file()
    x_in_train, x_out_train, x_in_test, y_test = pp.preprocess(x_data, y_data)
    ae, encoder = get_model()

    ae.fit(x=[x_in_train, x_in_train, x_in_train], y=x_out_train, batch_size=batch_size, epochs=epochs)  # 150
    return encoder, x_in_test, y_test
