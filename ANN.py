import numpy
import scipy.special

class neuralnetwork:
    def __init__(self, nodes_list, learningrate, layer):
        #变量初始化
        self.weights_list = [] #i-1个
        self.nodes_list = nodes_list #i个
        self.lr = learningrate
        self.layer = layer
        #初始化权重
        for i in range (self.layer - 1):
            self.weights_list.append (numpy.random.rand(self.nodes_list[i + 1], self.nodes_list[i]) - 0.5)
        #激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        inputs_list = [] #i个
        outputs_list = [] #i-1个
        errors_list = [] #i-1个
        #第0层的输入
        inputs_list.append (numpy.array(inputs, ndmin = 2).T)
        #目标
        targets_list = numpy.array(targets, ndmin = 2).T
        #输入输出计算
        inputs_list.append (numpy.dot(self.weights_list[0], inputs_list[0]))
        outputs_list.append (self.activation_function (inputs_list[1]))
        for i in range (self.layer - 2):
            inputs_list.append (numpy.dot(self.weights_list[i+1], outputs_list[i])) #
            outputs_list.append (self.activation_function(inputs_list[i+2]))
        #误差计算   从后往前
        #error是反的！！！
        errors_list.append (targets_list - outputs_list[self.layer - 2])  #错了
        for i in range (self.layer - 2):
            errors_list.append (numpy.dot(self.weights_list[self.layer - 2 - i].T, errors_list[i]))
        #权重更新
        for i in range (self.layer - 2):
            self.weights_list[self.layer - 2 - i] += self.lr * numpy.dot((errors_list[i] * outputs_list[self.layer - 2 - i] * (1 - outputs_list[self.layer - 2 - i])), numpy.transpose(outputs_list[self.layer - 3 - i]))
        self.weights_list[0] += self.lr * numpy.dot((errors_list[self.layer - 2] * outputs_list[0] * (1 - outputs_list[0])), numpy.transpose(inputs_list[0]))   #这行有问题
        #print (self.weights_list)



    def query(self, inputs):
        inputs_list = []
        outputs_list = []
        inputs_list.append (numpy.array(inputs, ndmin = 2).T)
        inputs_list.append (numpy.dot(self.weights_list[0], inputs_list[0]))
        outputs_list.append (self.activation_function(inputs_list[1]))
        for i in range (self.layer - 2):
            inputs_list.append (numpy.dot(self.weights_list[i + 1], outputs_list[i]))
            outputs_list.append (self.activation_function(inputs_list[i+2]))
        return outputs_list[layer - 2]
