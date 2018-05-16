
from models.ANN import attention_network_1
from data.reader import Data



def see_intermidiate_states(net, data):
    print(' ')


if __name__ == "__main__":
    data = Data()
    net = attention_network_1(data)

    see_intermidiate_states(net, data)



