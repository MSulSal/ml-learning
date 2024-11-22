import numpy as np

# activation function: f(x) = 1 / 1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
'''
A neural network with:
- 2 inputs
- a hidden layer with 2 neurons (h1, h2)
- an output layer with 1 neuron (o1)
Each neuron has the same weights and bias:
- w = [0, 1]
- b = 0
'''
class NeuralNetwork:
    def __init__(self):

        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        z1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        z2 = self.w3 * x[0] + self.w4 * x[0] + self.b2
        h1 = sigmoid(z1)
        h2 = sigmoid(z2)
        z3 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(z3)

        return o1
    
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset
    - all_y_trues is a numpy array with n elements
    Elements in all_y_trues correspond to those in data
    '''
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # feed forward
                z1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                z2 = self.w3 * x[0] + self.w4 * x[0] + self.b2
                h1 = sigmoid(z1)
                h2 = sigmoid(z2)
                z3 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(z3)
                y_pred = o1

                # calculate partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)

                # neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(z3)
                d_ypred_d_w6 = h2 * deriv_sigmoid(z3)
                d_ypred_d_b3 = deriv_sigmoid(z3)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(z3)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(z3)


                # neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(z1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(z1)
                d_h1_d_b1 = deriv_sigmoid(z1)


                # neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(z2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(z2)
                d_h2_d_b2 = deriv_sigmoid(z2)

                
                # update weights and biases
                # neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# define dataset
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6]
])

all_y_trues = np.array([
    1,
    0,
    0,
    1
])

# train neural network
network = NeuralNetwork()
network.train(data, all_y_trues)

# make some predictions
emily = np.array([-7, -3])
frank = np.array([20, 2])
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
