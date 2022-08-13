import evaluation as eva
import model


if __name__ == '__main__':
    encoder, x_test, y_test = model.train(batch_size=64, epochs=150)
    result = eva.evaluation(encoder, x_test, y_test, top=3, Simi="cos")
    print(result)

