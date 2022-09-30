from stadle import AdminAgent
from stadle.lib.util import client_arg_parser
from stadle.lib.entity.model import BaseModel
from stadle import BaseModelConvFormat

from tensorflow import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from stadle import BaseModelConvFormat
from stadle.lib.entity.model import BaseModel



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

    return BaseModel("Tensorflow-Mnist-Model", model, BaseModelConvFormat.keras_format)


if __name__ == '__main__':
    args = client_arg_parser()

    admin_agent = AdminAgent(config_file="config/config_agent.json", simulation_flag=args.simulation,
                             aggregator_ip_address=args.aggregator_ip, reg_port=args.reg_port, agent_name=args.agent_name,
                             exch_port=args.exch_port, model_path=args.model_path, base_model=get_mnist_model(),
                             agent_running=False)

    admin_agent.preload()
    admin_agent.initialize()