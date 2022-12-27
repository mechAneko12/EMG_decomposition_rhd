import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("fr_1031ff.csv", header=None)
print("ClassShape", df.shape)
fr_data = df.values
print(fr_data.shape)

sc = MinMaxScaler(feature_range = (0, 1))

#データを説明変数と目的変数にわける
data = fr_data[:,0:-1]
target = fr_data[:,-1]

#データ数の確認
data.shape

# tensor形式へ変換
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.int64)
target = target.long()

# 目的変数と入力変数をまとめてdatasetに変換
dataset = torch.utils.data.TensorDataset(data,target)



# 然后定义下标indices_val来表示校验集数据的那些下标，indices_test表示测试集的下标
indices = range(len(data))
indices_train = indices[:170]
indices_val = indices[170:180]
indices_test = indices[180:]

# 根据这些下标，构造两个数据集的SubsetRandomSampler采样器，它会对下标进行采样
sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

# 各データセットのサンプル数を決定
#train : val : test = 60% : 20% : 20%
#n_train = int(len(dataset) * 0.6)
#n_val = int((len(dataset) - n_train) * 0.5)
#n_test = len(dataset) - n_train - n_val

# データセットの分割
#torch.manual_seed(0) #乱数を与えて固定
#train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val,n_test])
#train, val, test = dataset[0:400,:], dataset[400:850,:], dataset[850:1248,:]

#バッチサイズ
batch_size = 32

# 乱数のシードを固定して再現性を確保
torch.manual_seed(0)

# shuffle はデフォルトで False のため、学習データのみ True に指定
train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler = sampler_train)
val_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler = sampler_train)
test_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler = sampler_train)

# 辞書型変数にまとめる(trainとvalをまとめて出す)
dataloaders_dict = {"train": train_loader, "val": val_loader}

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18, 18)
        self.fc2 = nn.Linear(18, 4)

    # 順伝播
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# インスタンス化
net = Net()

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法の選択
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):

    #入れ子を用意(各lossとaccuracyを入れていく)
    l= []
    a =[]


    # ベストなネットワークの重みを保持する変数
    best_acc = 0.0

    # GPUが使えるのであればGPUを有効化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # (エポック)回分のループ
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)

        for phase in ['train', 'val']:

            if phase == 'train':
                # 学習モード
                net.train()
            else:
                # 推論モード
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # 第1回で作成したDataLoaderを使ってデータを読み込む
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 勾配を初期化する
                optimizer.zero_grad()

                # 学習モードの場合のみ勾配の計算を可能にする
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    # 損失関数を使って損失を計算する
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # 誤差を逆伝搬する
                        loss.backward()
                        # パラメータを更新する
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # 1エポックでの損失を計算
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            # 1エポックでの正解率を計算
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            #lossとaccをデータで保存する
            a_loss = np.array(epoch_loss)
            a_acc = np.array(epoch_acc.cpu()) #GPU⇒CPUに変換してnumpy変換
            a.append(a_acc)
            l.append(a_loss)


            # 一番良い精度の時にモデルデータを保存
            if phase == 'val' and epoch_acc > best_acc:
                print('save model epoch:{:.0f} loss:{:.4f} acc:{:.4f}'.format(epoch,epoch_loss,epoch_acc))
                torch.save(net, 'best_model.pth')


    #testとvalのlossとaccを抜き出してデータフレーム化
    a_train = a[::2]
    l_train = l[::2]
    a_train = pd.DataFrame({'train_acc':a_train})
    l_train = pd.DataFrame({'train_loss':l_train})

    a_val = a[1::2]
    l_val = l[1::2]
    a_val = pd.DataFrame({'val_acc':a_val})
    l_val = pd.DataFrame({'val_loss':l_val})

    df_acc = pd.concat((a_train,a_val),axis=1)
    df_loss = pd.concat((l_train,l_val),axis=1)

    #ループ終了後にdfを保存
    df_acc.to_csv('acc.csv', encoding='shift_jis')
    df_loss.to_csv('loss.csv', encoding='shift_jis')

#学習と検証
num_epoch = 20
net = train_model(net, dataloaders_dict, criterion, optimizer, num_epoch)

#最適なモデルを呼び出す
best_model = torch.load('best_model.pth')

ceshiji = torch.from_numpy(fr_data[120:, 0:-1].astype(np.float32))
y_test = fr_data[120:, -1]


def evaluateModel(prediction, y):
    prediction = torch.argmax(prediction, dim=1)
    #     y = torch.argmax(y, dim=1)
    good = 0
    for i in range(len(y)):
        if (prediction[i] == y[i]):
            good = good + 1
    return (good / len(y)) * 100.0


with torch.no_grad():
    # y_hat_train = model(X_train)
    # print("Train Accuracy ", evaluateModel(y_hat_train, y_train))

    y_hat_test = best_model(ceshiji)
    print("Test Accuracy ", evaluateModel(y_hat_test, y_test))
    yuce = torch.argmax(y_hat_test, dim=1)
    print(yuce)
    print(y_test)
    
# 正解率の計算
# def test_model(test_loader):

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     with torch.no_grad():

#         accs = [] # 各バッチごとの結果格納用

#         for batch in test_loader:
#             x, t = batch
#             x = x.to(device)
#             t = t.to(device)
#             y = best_model(x)

#             y_label = torch.argmax(y, dim=1)
#             #print(y, y_label)
#             acc = torch.sum(y_label == t) * 1.0 / len(t)
#             accs.append(acc)

#     # 全体の平均を算出
#     avg_acc = torch.tensor(accs).mean()
#     std_acc = torch.tensor(accs).std()
#     print('Accuracy: {:.1f}%'.format(avg_acc * 100))
#     print('Std: {:.4f}'.format(std_acc))

# # テストデータで結果確認
# test_model(test_loader)

# torch.save(best_model, "drive/MyDrive/code/data/fr_grasp.csv")