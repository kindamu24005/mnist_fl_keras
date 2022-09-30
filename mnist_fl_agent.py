from tensorflow import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


from stadle import BasicClient

import argparse

from sklearn.model_selection import train_test_split


# データのロード
def load_MNIST(trainsize):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=trainsize, random_state=2)

    return x_train, x_valid, y_train, y_valid


# モデルの作成
def get_mnist_model():

    model = Sequential()

    model.add(Conv2D(filters = 20, kernel_size = (5, 5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters = 50, kernel_size = (5, 5), padding = 'Same', activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(500,kernel_size =  (4, 4),padding = 'Same', activation ='relu'))
    model.add(Conv2D(10,kernel_size =  (1, 1),padding = 'Same', activation ='relu'))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# 学習
def train(self, _x_train, _y_train, _x_valid, _y_valid, cb_origin):
    hist = self.model.fit(
        _x_train,
        _y_train, 
        validation_data=(_x_valid, _y_valid), 
        epochs=self.epoch, 
        batch_size=self.batch_size, 
        verbose=1,
        callbacks=[cb_origin])

    return hist

# オリジナルのHistoryを作成
class OriginHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        
        # Epochが偶数の時のみ処理
        if (epoch % 2 == 0):

            # 最初のEpochは処理しない
            if (e != 0):

                # 偶数Epoch終了毎に値を上のリストに入れる   
                pref_dict = {         
                    self.logs['acc'],
                    self.logs['val_acc'],
                    self.logs['loss'],
                    self.logs['val_loss']
                }

                # パフォーマンス値とモデルを送信
                stadle_client.send_trained_model(model, perf_dict)
            
            # 集約されたグローバルモデルをサーバーから取得
            state_dict = stadle_client.wait_for_sg_model().state_dict()
            model.load_state_dict(state_dict)
            




# Basic clientの作成
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')
    parser.add_argument('--agent_name', default='default_agent')
    parser.add_argument('--datasize', default=50000)
    parser.add_argument('--reg_port')
    args = parser.parse_args()

    # epoch,　batch_sizeの決定
    epoch = 100
    batch_size = 8

    client_config_path = r'config/config_agent.json'

    # モデルの取得
    model = get_mnist_model()

    # データの取得
    x_train, x_valid, y_train, y_valid = load_MNIST(trainsize=args.trainsize)

    # Preload stadle_client
    stadle_client = BasicClient(config_file=client_config_path, agent_name=args.agent_name, reg_port=args.reg_port)
    stadle_client.set_bm_obj(model)

    # callbackのインスタンスを作成
    cd_origin = OriginHistory()

    # 学習
    hist = model.train(x_train, y_train, x_valid, y_valid, cd_origin)