# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
# from dataset.mnist import load_mnist
# from common.functions import sigmoid, softmax
from scipy.special import expit  # expitはシグモイド関数

from sklearn import datasets
from sklearn.model_selection import train_test_split
def softmax(x):
    x = x - np.max(x)  # オーバーフロー対策
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def get_data():
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255  # ピクセル値を0～1に正規化
    y = y.astype('int64')  # ラベルを整数型に
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 例えば20%をテストデータに
        stratify=y,
        random_state=0
    )
    return X_test, y_test
'''
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
"""
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X, y = X.values, y.values
X_train, X_test, y_train, y_test = train_test_split(X / 255, # ピクセル値が 0 - 1 になるようにする
                                                    y.astype('int64'), # 正解データを数値にする
                                                    stratify = y,
                                                    random_state=0)
"""
'''
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))