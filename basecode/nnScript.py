import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle
import matplotlib.pyplot as plt


def initializeWeights(n_in, n_out):
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    
    return (1.0 / (1.0 + np.exp(-z)))
    
def rearrange_data(temp_data, temp_label):
    
    size = range(temp_data.shape[0])
    perm = np.random.permutation(size)
    data = temp_data[perm]
    data = data / 255.0
    label = temp_label[perm]
    return data, label
    
def forward_pass(training_data_i, W1, W2):
    
    net_ip = np.dot(training_data_i, W1.T)
    
    hidden_op = sigmoid(net_ip)
    
    hidden_op_i = np.concatenate((np.ones((hidden_op.shape[0],1)), hidden_op), axis=1)
    
    net_op = np.dot(hidden_op_i, W2.T)
    
    output = sigmoid(net_op)
    
    return hidden_op_i, output


def preprocess():

    mat = loadmat('mnist_all.mat')

    temp_train = np.zeros(shape=(50000, 784))
    temp_validation = np.zeros(shape=(10000, 784))
    temp_test = np.zeros(shape=(10000, 784))

    temp_train_label = np.zeros(shape=(50000,))
    temp_validation_label = np.zeros(shape=(10000,))
    temp_test_label = np.zeros(shape=(10000,))

    train_length = validation_length = test_length = train_label_length = validation_label_length = 0

    i = 0
    while i < 10:
        index = 'train'+ str(i)
        train_mat = mat[index]

        tag_length = len(train_mat) - 1000 

        permutation = np.random.permutation(range(train_mat.shape[0]))
        temp_train[train_length:train_length + tag_length] = train_mat[permutation[1000:], :]
        train_length += tag_length

        temp_train_label[train_label_length:train_label_length + tag_length] = i
        train_label_length += tag_length

        temp_validation[validation_length:validation_length + 1000] = train_mat[permutation[0:1000], :]
        validation_length += 1000

        temp_validation_label[validation_label_length:validation_label_length + 1000] = i
        validation_label_length += 1000
  
        index = 'test'+ str(i)
        test_mat = mat[index]

        tup_length = len(test_mat)

        permutation = np.random.permutation(range(test_mat.shape[0]))
        temp_test_label[test_length:test_length + tup_length] = i
        temp_test[test_length:test_length + tup_length] = test_mat[permutation]
        test_length += tup_length
        
        i +=1
        
    train_data, train_label = rearrange_data(temp_train, temp_train_label) 
    validation_data, validation_label = rearrange_data(temp_validation, temp_validation_label)
    test_data, test_label = rearrange_data(temp_test, temp_test_label)
    

    # Feature selection
    
    unsplit_data  = np.concatenate((train_data, validation_data),axis=0)
    copy_data = unsplit_data[0,:]
    duplicate_values = np.all(unsplit_data == copy_data, axis = 0)
    
    used_index_values = []
    
    for i in range(len(duplicate_values)):
        if duplicate_values[i]==False:
            used_index_values.append(i)
    
    final = unsplit_data[:,~duplicate_values]

    train_data      = final[0:train_data.shape[0],:]
    validation_data = final[train_data.shape[0]:,:]
    test_data = test_data[:,~duplicate_values]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, used_index_values

def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    n_count = training_data.shape[0]
    
    training_data_i = np.concatenate((np.ones((training_data.shape[0],1)), training_data), axis=1)
    
    hidden_op_i,output_data = forward_pass(training_data_i, w1, w2)
    
    label_output = np.zeros((n_count, n_class), dtype=int)
        
        
    size = np.size(training_label)
    label_output = np.zeros((size, n_class), dtype=int)
    for i in range(n_count):
        index = int(training_label[i])
        label_output[i][index] = 1
        
    label_output.T

    term1 = np.log(1-output_data)
    term2 = 1-label_output
    
    sum_term = np.sum(np.multiply(label_output,np.log(output_data)) + np.multiply(term2,term1))
    error = sum_term/(-n_count)
    
    delta = output_data- label_output
    w2_gradient = np.dot(delta.T, hidden_op_i)
   
    term11 = np.dot(delta,w2) * (hidden_op_i * (1.0-hidden_op_i))
    
    w1_gradient = np.dot( term11.T, training_data_i)
    w1_gradient = w1_gradient[1:, :]
    
    regularization =  lambdaval * (np.sum(w1**2) + np.sum(w2**2)) / (2*n_count)
    obj_val = error + regularization
    
    w1_gradient_reg = (w1_gradient + lambdaval * w1)/n_count
    w2_gradient_reg = (w2_gradient + lambdaval * w2)/n_count

    obj_grad = np.concatenate((w1_gradient_reg.flatten(), w2_gradient_reg.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, training_data):

    training_data_i = np.concatenate((np.ones((training_data.shape[0],1)), training_data), axis=1)
    
    hidden_op_i,output_data = forward_pass(training_data_i, w1, w2)

    labels = labels = np.argmax(output_data, axis=1)

    return labels



train_data, train_label, validation_data, validation_label, test_data, test_label, featureIndices = preprocess()

print(featureIndices)
print('Total Indices: '+str(len(featureIndices)))


n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


details = {
    
            "selected features": featureIndices,
            "n hidden": n_hidden,
            "w1": w1,
            "w2": w2,
            "lambda": lambdaval
            }

pickle.dump( details, open( "params.pickle", "wb" ) )

details = pickle.load( open( "params.pickle", "rb" ) )
print(details)
