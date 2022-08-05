# deepneuron

This is a package to perform deep learning.

### Installation:

    pip install deepneuron

### Implementation:

    from deepneuron import NeuralNetwork

    class NeuralNetwork:

        def __init__(self, 
                    layer_dims, 
                    layers, activations, 
                    cost_function, 
                    alpha0=0.003, 
                    lambd=0.1, 
                    batch_size=64, 
                    keep_prob=None, 
                    beta1=0.9, 
                    beta2=0.999, 
                    decay_rate=0, 
                    decay_interval=100, 
                    epsilon=1e-8, 
                    scale_input=False, 
                    batch_norm=None, 
                    bn_momentum=0.99, 
                    grad_check=False)
    
        def fit(self, 
                X, 
                Y, 
                num_epochs=100, 
                init_params=False)
    
        def predict(self, 
                    X, 
                    Y=None, 
                    batch_size=None)

### Parameters:

layer_dims: List of dimensions of all layers including the input 
dimensions. eg: [(28, 28, 1), ((3,3), (2,2), 2, 5), ((2,2), 2), 5, 10]. 
For the input, convolution and pooling layers, value should be a tuple.
(28, 28, 1) is the input dimensions. For a convolution later, the dimensions will be
(filter size, stride, padding, no of filters), so a layer with dimensions
((3,3), (2,2), 2, 5) means filter size is (3,3), stride is (2,2), padding 
is 2 ad it has 5 filters. For a max or avg pooling layer, the dimensions 
will be (stride, padding), so ((2,2), 2) means stride is (2,2) and padding 
is 2. For dense layers, the value will be the no. of neurons in the layer.

layers: List containing the types of layers. eg: layers = ["conv", "avgpool", "dense", "dense"]. 
The types of layers are "conv", "avgpool", "maxpool" and "dense".

activations: List of activations of the layers. eg: ["relu", "linear", "relu", "softmax"]. 
The types of activations are "relu", "sigmoid", "tanh", "linear" and "softmax". 
The output layer should have either "sigmoid" or "softmax" activations.

cost_function: The type of loss function. It can be "binary_cross_entropy" or 
"categorical_cross_entropy".

alpha0: The learning rate of the model. Its value is 
alpha = alpha0 / (1 + decay_rate * (iteration // decay_interval))

lambd: The L2 regularization parameter of the model.

batch_size: The size of each batch.

keep_prob: It is a list of floats for each layer. eg: [0.9, 0.9, 0.9, 1]. 
The probability of keeping an activation for dropout regularization.

beta1: The first moment of the Adam optimization algorithm.

beta2: The second moment of the Adam optimization algorithm.

decay_rate: The rate at which the learning rate should decay.

decay_interval: No. of epochs after which learning rate should decay.

epsilon: Add to variance during normalization.

scale_input: Boolean value to indicate whether model should normalize inputs.

batch_norm: It is a list of booleans for each layer. eg: [True, True, True, False]. 
It indicates whether model should perform batch normalization at that layer. 

bn_momentum: The momentum of the batch mean and variance when performing batch normalization. 

grad_check: Boolean value to indicate whether model should perform gradient checking. 
Used for testing.