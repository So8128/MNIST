# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import pickle
from mnist import load_mnist
from three_layer_net import ThreeLayerNet   

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 3層ネットワーク用 (例: 784→100→50→10)
network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    grad = network.gradient(x_batch, t_batch)
    
    # --- 3層用：W3, b3も追加 ---
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train_acc: {train_acc}, test_acc: {test_acc}")

# --- 保存処理 ---
with open("trained_weights.pkl", "wb") as f:
    pickle.dump(network.params, f)

np.save("train_loss_list.npy", np.array(train_loss_list))
np.save("train_acc_list.npy", np.array(train_acc_list))
np.save("test_acc_list.npy", np.array(test_acc_list))

print("重み・損失・精度を保存しました。")