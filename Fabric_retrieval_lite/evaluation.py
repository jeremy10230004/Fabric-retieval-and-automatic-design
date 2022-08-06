import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2


def Euclidean_Distance(a, b_list):
    ans = []

    for b in b_list:
        ans.append(np.sqrt(np.sum(np.square(a - b))))

    return np.array(ans)


def evaluation(model, x_test, y_test, top=3, Simi="cos"):
    x_test_o = [cv2.resize(x, (64, 64)) for x in np.load('x_data.npy')]
    x_test_o = np.array(x_test_o) / 255
    y_test_o = np.load('y_data.npy')

    # ---------------------------------------
    features = model.predict(x_test)
    features_o = model.predict(x_test_o)

    pred_l = []
    for id, f in enumerate(features):
        r = True
        if Simi == "cos":
            sim = cosine_similarity([f], features_o)[0]
        elif Simi == "eu":
            sim = Euclidean_Distance(f, features_o)
            r = False
        else:
            sim = []
            print('"sim" has to be "cos" or "eu"')

        sim = list(sim)
        sim_sort = sorted(sim, reverse=r)
        ans = []
        for i in range(top):
            ans.append(sim.index(sim_sort[i]))
            sim[sim.index(sim_sort[i])] = 0
        pred_l.append(y_test_o[ans])

    sum_ = 0
    for id_, c in enumerate(pred_l):
        # print(y_test[id_], pred_l[id_])
        if y_test[id_] in pred_l[id_]:
            sum_ += 1
        else:
            sum_ += 0

    return sum_ / len(pred_l)
