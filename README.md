# deepneuron

This is a package to perform deep learning.

Below is the implementation of the module.

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