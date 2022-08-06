import numpy as np
import random
from copy import deepcopy
import cv2
from sklearn.model_selection import train_test_split


def load_file():
    return np.load("x_data.npy"), np.load("y_data.npy")


def preprocess(x_data, y_data, cut_amount=10, hue_amount=3, result_size=(64, 64), test_size=0.2):
    x_ans = []

    for x in x_data:
        for i in range(cut_amount):
            r1 = random.randint(0, 100)
            r2 = random.randint(0, 100)
            new_x = x[r1:r1 + 200, r2:r2 + 200]
            x_ans.append(new_x)

    x_ans2 = []
    for x in x_ans:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2HLS)
        for i in range(hue_amount):
            new_x = deepcopy(x)
            r = random.randint(-15, 15)
            new_x[:, :, 0] = new_x[:, :, 0] + r
            new_x = cv2.cvtColor(new_x, cv2.COLOR_HLS2RGB)
            x_ans2.append(new_x)

    x_ans2 = np.array(x_ans2)
    x_final = []
    for x in x_ans2:
        x_final.append(cv2.resize(x, result_size))

    x_final = np.array(x_final)

    # -----------------------------------------
    x_final_out = []
    for x in x_data:
        for i in range(cut_amount*hue_amount):
            x_final_out.append(cv2.resize(x, result_size))

    x_final_out = np.array(x_final_out)
    # -----------------------------------------
    y_final = []
    for y in y_data:
        for i in range(cut_amount*hue_amount):
            y_final.append(y)

    y_final = np.array(y_final)

    x_in_train, x_in_test, x_out_train, _, _, y_test = train_test_split(x_final, x_final_out, y_final, test_size=test_size)
    x_in_train, x_in_test = x_in_train / 255, x_in_test / 255
    x_out_train = x_out_train / 255

    return x_in_train, x_out_train, x_in_test, y_test
