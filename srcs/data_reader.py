import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import natsort
import sys

class DataReader():
    def __init__(self):
        self.label = ["covid", "normal", "pneumonia", "tuberculosis"]  # 코로나, 일반, 폐렴, 결핵
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        self.read_images()


    def read_images(self):
        print("Reading Data...")
        classes = os.listdir("../train_data")
        classes = natsort.natsorted(classes)
        for i, cls in enumerate(classes):  # i: label
            cls_dir = os.listdir("../train_data/" + cls)
            for j, c_dir in enumerate(cls_dir):  # j: train/test
                for el in os.listdir("../train_data/" + cls + "/" + c_dir):
                    filename = "../train_data/" + cls + "/" + c_dir + "/" + el
                    if j == 0:  # test 디렉토리 안의 파일이라면
                        self.test_X.append(np.asarray(img_convert(filename)))
                        self.test_Y.append(i)
                    else:  # train 디렉토리 안의 파일이라면
                        self.train_X.append(np.asarray(img_convert(filename)))
                        self.train_Y.append(i)

        # normalize
        self.train_X = np.asarray(self.train_X) / 255.0
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X) / 255.0
        self.test_Y = np.asarray(self.test_Y)

        # 흑백이미지를 학습시키기 위해 의도적으로 차원을 하나 추가해 줌
        # https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4
        self.train_X = self.train_X.reshape(self.train_X.shape[0], 256, 256, 1)
        self.test_X = self.test_X.reshape(self.test_X.shape[0], 256, 256, 1)

        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')


def img_convert(filename):
    try:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 사진으로 변환
        img = cv2.resize(img, (256, 256))  # 256 * 256 사이즈로 변환
    except:
        print(filename, ": 이미지 파일이 맞는지 확인해 주세요.", file=sys.stderr)
    return img


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("../results/train_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("../results/accuracy_history.png")