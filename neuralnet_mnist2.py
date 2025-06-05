'''
# coding: utf-8
import sys, os
import pickle
from mnist import load_mnist
from three_layer_net import ThreeLayerNet 
# ====== データの読み込み（学習時と同じ前処理！） ======
(_, _), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ====== ネットワークの準備（学習時と同じ構造・引数） ======
network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)
with open("trained_weights.pkl", "rb") as f:
    params = pickle.load(f)
network.params = params

# ====== テストデータで精度を評価 ======
accuracy = network.accuracy(x_test, t_test)
print(f"Test Accuracy: {accuracy:.4f}")
for key in network.params:
    print(f"{key}: mean={network.params[key].mean():.6f}, std={network.params[key].std():.6f}")
    '''
# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 'common' フォルダが親ディレクトリにある場合に必要
import pickle
from mnist import load_mnist
from three_layer_net import ThreeLayerNet
# from common.layers import Affine # Affineクラスのインポートは不要 (ThreeLayerNet内部で使われるため)

# ====== データの読み込み（学習時と同じ前処理！） ======
(_, _), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ====== ネットワークの準備（学習時と同じ構造・引数） ======
network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)

with open("trained_weights.pkl", "rb") as f:
    loaded_params = pickle.load(f)

# network.params にロードしたパラメータを代入 (これは統計情報表示などのためには良い)
network.params = loaded_params

# ▼▼▼▼▼ 重要: Affineレイヤの重みをロードした値で更新 ▼▼▼▼▼
network.layers['Affine1'].W = network.params['W1']
network.layers['Affine1'].b = network.params['b1']
network.layers['Affine2'].W = network.params['W2']
network.layers['Affine2'].b = network.params['b2']
network.layers['Affine3'].W = network.params['W3']
network.layers['Affine3'].b = network.params['b3']
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ====== テストデータで精度を評価 ======
accuracy = network.accuracy(x_test, t_test)
print(f"Test Accuracy: {accuracy:.4f}")

# パラメータの統計情報も表示 (ロードした値が反映されているはず)
for key in network.params:
    # network.params[key] が NumPy 配列であることを想定
    print(f"{key}: mean={network.params[key].mean():.6f}, std={network.params[key].std():.6f}")