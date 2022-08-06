import Data_Preprocess as pp
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import evaluation as eva


def get_model():
    input_layers = Input((64, 64, 3))
    en = Conv2D(32, (4, 4), padding='same', strides=2, activation='relu')(input_layers)
    en = Conv2D(16, (4, 4), padding='same', strides=2, activation='relu')(en)
    en = Conv2D(16, (4, 4), padding='same', strides=2, activation='relu')(en)
    f = Flatten()(en)
    d = Dense(8 * 8 * 8, activation='relu')(f)
    feature = Dense(256, activation='sigmoid')(d)
    d = Dense(8 * 8 * 8, activation='relu')(feature)
    r = Reshape((8, 8, 8))(d)
    de = Conv2DTranspose(16, (4, 4), padding='same', strides=2, activation='relu')(r)
    de = Conv2DTranspose(16, (4, 4), padding='same', strides=2, activation='relu')(de)
    de = Conv2DTranspose(32, (4, 4), padding='same', strides=2, activation='relu')(de)
    output_layers = Conv2DTranspose(3, (4, 4), padding='same', activation='sigmoid')(de)

    ae = Model(input_layers, output_layers)
    ae.compile(optimizer='adam', loss='mse')
    ae.summary()

    encoder = Model(input_layers, feature)
    encoder.compile()
    encoder.summary()
    return ae, encoder


def train(batch_size=64, epochs=3):
    x_data, y_data = pp.load_file()
    x_in_train, x_out_train, x_in_test, y_test = pp.preprocess(x_data, y_data)
    ae, encoder = get_model()

    ae.fit(x=x_in_train, y=x_out_train, batch_size=batch_size, epochs=epochs)  # 150
    return encoder, x_in_test, y_test


if __name__ == '__main__':
    encoder, x_test, y_test = train(batch_size=64, epochs=100)
    result = eva.evaluation(encoder, x_test, y_test, top=3, Simi="eu")
    print(result)

