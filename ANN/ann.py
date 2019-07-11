import numpy as np

class ArtificialNeuralNetwork():

    ''' A simple neural network with a Single hidden layer.
        Number of units in the layer can be passed while declaring the object of this class
        Weights of the both hidden layer and output layer are initialized using standard normal distribution
        Bias at the hidden layer and output layer are considered 0.
    '''

    global hidden_layer_Z, hidden_layer_A, output_layer_Z, output_layer_A

    def __init__(self, features, output, number_of_units):
        self.number_of_units = number_of_units
        self.number_of_samples, self.number_of_features = features.shape
        self.features = features.reshape(self.number_of_features, self.number_of_samples)
        self.output = output.reshape(1, self.number_of_samples)
        self.hidden_layer_weights = np.random.randn(self.number_of_units, self.number_of_features)* 0.01
        self.output_layer_weights = np.random.randn(1, self.number_of_units) * 0.01

    def relu(self,Z):
        A = np.maximum(0, Z)
        return A

    def sigmoid(self,Z):
        A = 1/(1 + np.exp(-Z))
        return A

    def reluDerivative(self,Z):
        rows, columns = Z.shape
        list_of_derivatives = []
        for value in np.nditer(Z):
            if value < 0:
                list_of_derivatives.append(0)
            else:
                list_of_derivatives.append(1)
        return np.array(list_of_derivatives).reshape(rows, columns)

    def fit(self, number_of_epochs, learning_rate):
        for i in range(number_of_epochs):
            #forward propagation
            self.forwardPropagation(self.features)
            #backPropagation
            self.backPropagation(learning_rate)

    def forwardPropagation(self, input_data):
        self.hidden_layer_Z = np.dot(self.hidden_layer_weights, input_data)
        self.hidden_layer_A = self.relu(self.hidden_layer_Z)
        self.output_layer_Z = np.dot(self.output_layer_weights, self.hidden_layer_A)
        self.output_layer_A = self.sigmoid(self.output_layer_Z)

    def backPropagation(self, learning_rate):
        dZ_output_layer = self.output_layer_A - self.output
        dW_output_layer = (1/self.number_of_samples) * np.dot(dZ_output_layer, self.hidden_layer_A.T)
        self.output_layer_weights = self.output_layer_weights - (dW_output_layer * learning_rate)
        dZ_hidden_layer = np.multiply(np.dot(dW_output_layer.T, dZ_output_layer), self.reluDerivative(self.hidden_layer_Z))
        dW_hidden_layer = (1/self.number_of_samples) * np.dot(dZ_hidden_layer, self.features.T)
        self.hidden_layer_weights = self.hidden_layer_weights - (dW_hidden_layer * learning_rate)

    def predict(self, test_set):
        number_of_test_samples, number_of_test_features = test_set.shape
        test_set = test_set.reshape(number_of_test_features, number_of_test_samples)
        self.forwardPropagation(test_set)
        return self.output_layer_A.reshape(number_of_test_samples, 1)
