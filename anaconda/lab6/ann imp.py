import numpy as np
import time

input_data=np.array([[0,0],[1,0],[0,1],[1,1]])
output_data=np.array([[0],[1],[1],[1]])
input_layer_num=2
hidden_layer_num=3
output_layer_num=1
input_layer_weight=np.random.uniform(size=(input_layer_num,hidden_layer_num))
input_layer_bias=np.random.uniform(size=(1,hidden_layer_num))
output_layer_weight=np.random.uniform(size=(hidden_layer_num,output_layer_num))
output_layer_bias=np.random.uniform(size=(1,output_layer_num)) 
print('the input is',input_data.shape)
print('the output is',output_data.shape)
print('the input weight is ',input_layer_weight.shape)
print('the input bias is ',input_layer_bias.shape)
print('the output weight is ',output_layer_weight.shape)
print('the output bias is ',output_layer_bias.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    return x*(1-x)

lr=0.1

start = time.time()

for i in range(100000):
    hidden_layer_activation=np.dot(input_data,input_layer_weight)
    hidden_layer_activation+=input_layer_bias
    hidden_layer_sigmoid=sigmoid(hidden_layer_activation)

    output_layer_activation=np.dot(hidden_layer_sigmoid,output_layer_weight)
    output_layer_activation+=output_layer_bias
    predicted_layer=sigmoid(output_layer_activation)

    error_end=output_data-predicted_layer

    derivative_end=error_end*derivative(predicted_layer)
    error_hidden_layer=derivative_end.dot(output_layer_weight.T)
    derivative_hidden=error_hidden_layer*derivative(hidden_layer_sigmoid)

    output_layer_weight+=hidden_layer_sigmoid.T.dot(derivative_end)*lr
    output_layer_bias+=np.sum(derivative_end,axis=0,keepdims=True)*lr
    input_layer_weight+=input_data.T.dot(derivative_hidden)*lr
    input_layer_bias+=np.sum(derivative_hidden,axis=0,keepdims=True)*lr

    

end = time.time()

print(predicted_layer)
print('The time taken -- {}'.format(end-start))
