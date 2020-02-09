import numpy as np
import math
import time

class Layer:
    def __init__(self, node_num, act_func, deri_act_func):
        self.node_num = node_num
        self.act_func = act_func
        self.deri_act_func = deri_act_func

    def compile_layer(self, prev_node_num):
        self.weight_mat = np.random.random(size = (prev_node_num, self.node_num))

    def forward(self, inputs):
        owa = np.dot(np.array(inputs, ndmin = 2), self.weight_mat)
        return [self.act_func(x) for x in owa[0]]

class Network:
    def __init__(self, layer_list, in_node_num):
        self.layer_list = layer_list
        self.in_node_num = in_node_num

    def compile_network(self):
        
        self.layer_list[0].compile_layer(self.in_node_num)
        for i in range(1, len(self.layer_list)):
            self.layer_list[i].compile_layer(self.layer_list[i - 1].node_num)

    def train(self, inputs, targets, learning_rate):
        inputs = np.array(inputs)
        targets = np.array(targets)
        output_list = []
        output_list.append(inputs)
        for i in range(len(self.layer_list)):
            output_list.append(self.layer_list[i].forward(output_list[i]))

        error_list = []
        error_list.append(targets - output_list[-1])
        for i in range(len(self.layer_list) - 1):
            error_list.append(np.dot(np.array(error_list[-1], ndmin = 2), self.layer_list[len(self.layer_list) - i - 1].weight_mat.T)[0])
        
        for i in range(len(self.layer_list)):
            delta_W = np.dot(
                np.array(output_list[i], ndmin = 2).T,
                np.array((error_list[len(error_list) - i - 1] * [self.layer_list[i].deri_act_func(x) for x in np.dot(np.array(output_list[i], ndmin = 2), self.layer_list[i].weight_mat)[0]]), ndmin = 2)
            )
            self.layer_list[i].weight_mat += learning_rate * delta_W  # Is plus, zheng fu di xiao
        # print(error_list)

    def query(self, inputs):
        for i in range(len(self.layer_list)):
            inputs = self.layer_list[i].forward(inputs)
        return inputs


def f(x):
    s = 1 / (1 + np.exp(-x))
    return s

def g(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

def a(x):
    # print('IN ', x)
    return x

def b(x):
    return 1
if __name__ == "__main__":
    l_i = [0, 0.2, 0.4, 0.6, 0.8, 1]
    l_t = [0, 0.04, 0.16, 0.36, 0.64, 1]
    A = Layer(2, f, g)
    B = Layer(1, f, g)

    nn = Network([A, B], 1)
    nn.compile_network()
    

    for i in range(100000):
        # l_o = []
        for j in range(len(l_i)):
            # l_o.append(nn.query([l_i[j]]))
            nn.train(l_i[j], l_t[j], 1)
        print(nn.query([0]), nn.query([0.2]), nn.query([0.4]), nn.query([0.6]), nn.query([0.8]), nn.query([1]))