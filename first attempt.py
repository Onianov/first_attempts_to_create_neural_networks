import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


input_data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
output_data = np.array([[1], [1], [0], [0]])

w1 = np.random.randn(2, 6)
w2 = np.random.randn(6, 1)


for i in range(100000):
    hidden_layer = sigmoid(np.dot(input_data, w1))
    output_layer = sigmoid(np.dot(hidden_layer, w2))

    loss = output_data - output_layer
    output_layer_delta = loss * output_layer * (1 - output_layer)
    hidden_layer_loss = output_layer_delta.dot(w2.T)
    hidden_layer_delta = hidden_layer_loss * hidden_layer * (1 - hidden_layer)

    w2 += hidden_layer.T.dot(output_layer_delta)
    w1 += input_data.T.dot(hidden_layer_delta)


new_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
test = sigmoid(np.dot(sigmoid(np.dot(new_data, w1)), w2))
print(test)
