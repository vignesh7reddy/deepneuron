import numpy as np


class NeuralNetwork:
    def __init__(self, layer_dims, layers, activations, cost_function, alpha0=0.003, lambd=0.1, batch_size=64 \
                 , keep_prob=None, beta1=0.9, beta2=0.999, decay_rate=0, decay_interval=100, epsilon=1e-8 \
                 , scale_input=False, batch_norm=None, bn_momentum=0.99, grad_check=False):
        self.__layer_dims = layer_dims
        self.__layers = layers
        self.__activations = activations
        self.__L = len(layer_dims) - 1
        self.__alpha0 = alpha0
        self.__lambd = lambd
        self.__batch_size = batch_size
        self.__keep_prob = keep_prob if keep_prob is not None else [1] * self.__L
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__decay_rate = decay_rate
        self.__decay_interval = decay_interval
        self.__epsilon = epsilon
        self.__scale_input = scale_input
        self.__batch_norm = batch_norm if batch_norm is not None else [False] * self.__L
        self.__bn_momentum = bn_momentum
        self.__grad_check = grad_check
        self.__initialize_parameters()
        self.__generate_activations()
        self.__generate_cost_function(cost_function)

    def fit(self, X, Y, num_epochs=100, init_params=False):
        self.__X = X
        self.__Y = Y
        self.__num_epochs = num_epochs
        self.__init_params = init_params
        if init_params:
            self.__initialize_parameters()
        self.__m = X.shape[0]
        self.__total_batches = self.__m // self.__batch_size
        if self.__m % self.__batch_size != 0:
            self.__total_batches += 1
        if self.__scale_input:
            self.__standardize_inputs()
        self.__gradient_descent()
        return self.__parameters, self.__cost_history

    def predict(self, X, Y=None, batch_size=None):

        if self.__scale_input:
            X = (X - self.__input_mean) / np.sqrt(self.__input_var + self.__epsilon)

        batch_size = X.shape[0] if batch_size is None else batch_size
        total_batches = X.shape[0] // batch_size
        total_batches = total_batches + 1 if X.shape[0] % batch_size != 0 else total_batches

        for batch_num in range(total_batches):
            if batch_num != total_batches - 1:
                X_minibatch = X[batch_num * batch_size: (batch_num + 1) * batch_size, :]
            else:
                X_minibatch = X[batch_num * batch_size:, :]

            for l in range(1, self.__L + 1):

                if self.__layers[l - 1] == "conv":
                    Z = self.__conv_forward(X_minibatch, self.__parameters[f"W{l}"], self.__parameters[f"b{l}"] \
                                            , self.__layer_dims[l][1], self.__layer_dims[l][2],
                                            self.__batch_norm[l - 1])

                    if self.__batch_norm[l - 1]:
                        Z = (Z - self.__batch_norm_params[f"V_mean{l}"]) \
                            / np.sqrt(self.__batch_norm_params[f"V_var{l}"] + self.__epsilon)
                        Z = Z * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]

                elif self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                    Z = self.__pool_forward(X_minibatch, self.__layer_dims[l][0], self.__layer_dims[l][1],
                                            self.__layers[l - 1])
                    if self.__batch_norm[l - 1]:
                        Z = (Z - self.__batch_norm_params[f"V_mean{l}"]) \
                            / np.sqrt(self.__batch_norm_params[f"V_var{l}"] + self.__epsilon)
                        Z = Z * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]

                else:
                    if self.__batch_norm[l - 1]:
                        Z = np.dot(X_minibatch, self.__parameters[f"W{l}"])
                        Z = (Z - self.__batch_norm_params[f"V_mean{l}"]) \
                            / np.sqrt(self.__batch_norm_params[f"V_var{l}"] + self.__epsilon)
                        Z = Z * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]
                    else:
                        Z = np.dot(X_minibatch, self.__parameters[f"W{l}"]) + self.__parameters[f"b{l}"]

                if self.__layers[l - 1] == "conv" or self.__layers[l - 1] == "maxpool" or self.__layers[
                    l - 1] == "avgpool":
                    if self.__layers[l] == "dense":
                        Z = np.reshape(Z, (Z.shape[0], -1))

                X_minibatch = self.__forward_activations[l - 1](Z)

                if l == self.__L:
                    if batch_num == 0:
                        X_pred = X_minibatch
                    else:
                        X_pred = np.concatenate((X_pred, X_minibatch), axis=0)

        if Y is not None:
            accuracy = np.mean(np.argmax(X_pred, axis=-1) == np.argmax(Y, axis=-1))
            print(f"Model accuracy: {accuracy}")
            return X_pred, accuracy
        return X_pred

    def get_outputs(self):
        return self.__outputs

    def get_parameters(self):
        return self.__parameters

    def set_parameters(self, parameters):
        self.__parameters = parameters

    def __standardize_inputs(self):
        self.__input_mean = np.mean(self.__X, axis=0, keepdims=True)
        self.__input_var = np.var(self.__X, axis=0, keepdims=True)
        self.__X = (self.__X - self.__input_mean) / np.sqrt(self.__input_var + self.__epsilon)

    def __initialize_parameters(self):
        self.__parameters = {}
        self.__batch_norm_params = {}
        self.__adam_parameters = {}
        self.__outputs = []
        self.__iteration = 0
        self.__cost_history = []
        self.__outputs.append(self.__layer_dims[0])
        # np.random.seed(0)
        for l in range(1, self.__L + 1):

            if self.__layers[l - 1] == "conv":
                # layerdims=((3,3), (2,2), 0, 32)
                self.__outputs.append(
                    ((self.__outputs[l - 1][0] + 2 * self.__layer_dims[l][2] - self.__layer_dims[l][0][0]) \
                     // self.__layer_dims[l][1][0] + 1 \
                         , (self.__outputs[l - 1][1] + 2 * self.__layer_dims[l][2] - self.__layer_dims[l][0][1]) \
                     // self.__layer_dims[l][1][1] + 1 \
                         , self.__layer_dims[l][3]))
                self.__parameters[f"W{l}"] = np.random.randn(self.__layer_dims[l][0][0], self.__layer_dims[l][0][1], \
                                                             self.__outputs[l - 1][2], self.__layer_dims[l][3]) * \
                                             np.sqrt(2 / self.__layer_dims[l][0][0] * self.__layer_dims[l][0][1] *
                                                     self.__outputs[l - 1][2])
                self.__adam_parameters[f"V_dW{l}"] = np.zeros(self.__parameters[f"W{l}"].shape)
                self.__adam_parameters[f"S_dW{l}"] = np.zeros(self.__parameters[f"W{l}"].shape)
                if self.__batch_norm[l - 1]:
                    self.__parameters[f"gamma{l}"] = np.ones((1,) + self.__outputs[l])
                    self.__adam_parameters[f"V_dgamma{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__adam_parameters[f"S_dgamma{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__parameters[f"b{l}"] = np.ones((1,) + self.__outputs[l])
                    self.__adam_parameters[f"V_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)
                    self.__adam_parameters[f"S_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)

                    self.__batch_norm_params[f"V_mean{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__batch_norm_params[f"V_var{l}"] = np.ones(self.__parameters[f"gamma{l}"].shape)
                else:
                    self.__parameters[f"b{l}"] = np.zeros((1, 1, 1, self.__layer_dims[l][3]))
                    self.__adam_parameters[f"V_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)
                    self.__adam_parameters[f"S_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)

            elif self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                self.__outputs.append(
                    ((self.__outputs[l - 1][0] + 2 * self.__layer_dims[l][1]) // self.__layer_dims[l][0][0] \
                         , (self.__outputs[l - 1][1] + 2 * self.__layer_dims[l][1]) // self.__layer_dims[l][0][1] \
                         , self.__outputs[l - 1][2]))
                if self.__batch_norm[l - 1]:
                    self.__parameters[f"gamma{l}"] = np.ones((1,) + self.__outputs[l])
                    self.__adam_parameters[f"V_dgamma{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__adam_parameters[f"S_dgamma{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__parameters[f"b{l}"] = np.ones((1,) + self.__outputs[l])
                    self.__adam_parameters[f"V_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)
                    self.__adam_parameters[f"S_db{l}"] = np.zeros(self.__parameters[f"b{l}"].shape)

                    self.__batch_norm_params[f"V_mean{l}"] = np.zeros(self.__parameters[f"gamma{l}"].shape)
                    self.__batch_norm_params[f"V_var{l}"] = np.ones(self.__parameters[f"gamma{l}"].shape)
            else:
                prev_outputs = np.prod(self.__outputs[l - 1])
                self.__outputs.append((self.__layer_dims[l],))
                self.__parameters[f"W{l}"] = np.random.randn(prev_outputs, self.__layer_dims[l]) * \
                                             np.sqrt(2 / prev_outputs)
                self.__parameters[f"b{l}"] = np.zeros((1, self.__layer_dims[l]))
                self.__adam_parameters[f"V_dW{l}"] = np.zeros((prev_outputs, self.__layer_dims[l]))
                self.__adam_parameters[f"V_db{l}"] = np.zeros((1, self.__layer_dims[l]))
                self.__adam_parameters[f"S_dW{l}"] = np.zeros((prev_outputs, self.__layer_dims[l]))
                self.__adam_parameters[f"S_db{l}"] = np.zeros((1, self.__layer_dims[l]))
                if self.__batch_norm[l - 1]:
                    self.__parameters[f"gamma{l}"] = np.ones((1, self.__layer_dims[l]))
                    self.__adam_parameters[f"V_dgamma{l}"] = np.zeros((1, self.__layer_dims[l]))
                    self.__adam_parameters[f"S_dgamma{l}"] = np.zeros((1, self.__layer_dims[l]))

                    self.__batch_norm_params[f"V_mean{l}"] = np.zeros((1, self.__layer_dims[l]))
                    self.__batch_norm_params[f"V_var{l}"] = np.ones((1, self.__layer_dims[l]))

    def __generate_activations(self):
        self.__forward_activations = []
        self.__backward_activations = []
        for activation in self.__activations:
            if (activation == "sigmoid"):
                self.__forward_activations.append(self.__sigmoid_forward)
                self.__backward_activations.append(self.__sigmoid_backward)
            if (activation == "relu"):
                self.__forward_activations.append(self.__relu_forward)
                self.__backward_activations.append(self.__relu_backward)
            if (activation == "tanh"):
                self.__forward_activations.append(self.__tanh_forward)
                self.__backward_activations.append(self.__tanh_backward)
            if (activation == "softmax"):
                self.__forward_activations.append(self.__softmax_forward)
            if (activation == "linear"):
                self.__forward_activations.append(self.__linear_forward)
                self.__backward_activations.append(self.__linear_backward)

    def __generate_cost_function(self, cost_function):
        if cost_function == "binary_cross_entropy":
            self.__cost_function = self.__compute_binary_cost
        elif cost_function == "categorical_cross_entropy":
            self.__cost_function = self.__compute_categorical_cost

    @staticmethod
    def __linear_forward(Z):
        A = Z
        return A

    @staticmethod
    def __sigmoid_forward(Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    @staticmethod
    def __relu_forward(Z):
        A = np.maximum(0, Z)
        return A

    @staticmethod
    def __tanh_forward(Z):
        A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return A

    @staticmethod
    def __softmax_forward(Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=-1, keepdims=True)
        return A

    @staticmethod
    def __linear_backward(A):
        return np.ones(A.shape)

    @staticmethod
    def __sigmoid_backward(A):
        return A * (1 - A)

    @staticmethod
    def __tanh_backward(A):
        return 1 - A ** 2

    @staticmethod
    def __relu_backward(A):
        return (A > 0).astype(int)

    @staticmethod
    def __dictionary_to_vector(dictionary):
        count = 0
        for key in sorted(dictionary):
            if count == 0:
                vector = dictionary[key].reshape(-1, )
            else:
                vector = np.concatenate((vector, dictionary[key].reshape(-1, )), axis=0)
            count += 1
        return vector

    @staticmethod
    def __vector_to_dictionary(vector, sample_dictionary):
        dictionary = {}
        count = 0
        for key in sorted(sample_dictionary):
            num_elements = sample_dictionary[key].size
            dictionary[key] = vector[count:count + num_elements].reshape(*sample_dictionary[key].shape)
            count += num_elements
        return dictionary

    def __gradient_check(self, cost_function, epsilon=1e-7):
        parameters_copy = self.__parameters.copy()
        layer_activations_copy = self.__layer_activations.copy()
        theta = self.__dictionary_to_vector(self.__parameters)
        dtheta = self.__dictionary_to_vector(self.__grads)
        dtheta_approx = np.zeros(dtheta.shape)
        num_parameters = theta.shape[0]

        for i in range(num_parameters):
            theta_plus = np.copy(theta)
            theta_plus[i] = theta_plus[i] + epsilon
            self.__parameters = self.__vector_to_dictionary(theta_plus, self.__parameters)
            self.__forward_prop(update=False)
            cost_plus = cost_function()

            theta_minus = np.copy(theta)
            theta_minus[i] = theta_minus[i] - epsilon
            self.__parameters = self.__vector_to_dictionary(theta_minus, self.__parameters)
            self.__forward_prop(update=False)
            cost_minus = cost_function()

            dtheta_approx[i] = (cost_plus - cost_minus) / (2 * epsilon)

        self.__parameters = parameters_copy
        self.__layer_activations = layer_activations_copy

        self.parameters = self.__parameters
        self.dtheta = dtheta
        self.dtheta_approx = dtheta_approx

        return np.linalg.norm(dtheta_approx - dtheta) / (np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta))

    def __compute_binary_cost(self):
        cost = np.mean(-self.__Y_minibatch * np.log(np.maximum(self.__layer_activations[f"A{self.__L}"], 1e-16)) \
                       - (1 - self.__Y_minibatch) * np.log(
            np.maximum(1 - self.__layer_activations[f"A{self.__L}"], 1e-16)))
        reg_cost = 0
        for l in range(1, self.__L + 1):
            if self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                continue
            reg_cost += np.sum(self.__parameters[f"W{l}"] ** 2)
        cost += self.__lambd * reg_cost / (2 * self.__curr_batch_size)
        return cost

    def __compute_categorical_cost(self):
        cost = np.sum(-self.__Y_minibatch * np.log(np.maximum(self.__layer_activations[f"A{self.__L}"], 1e-16))) \
               / self.__curr_batch_size
        reg_cost = 0
        for l in range(1, self.__L + 1):
            if self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                continue
            reg_cost += np.sum(self.__parameters[f"W{l}"] ** 2)
        cost += self.__lambd * reg_cost / (2 * self.__curr_batch_size)
        return cost

    @staticmethod
    def __pool_forward(a, stride, pad, mode):
        a = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))
        view_shape = (a.shape[0], (a.shape[1] - stride[0]) // stride[0] + 1 \
                          , (a.shape[2] - stride[1]) // stride[1] + 1, stride[0], stride[1], a.shape[3])
        strides = (a.strides[0], stride[0] * a.strides[1], stride[1] * a.strides[2]) + a.strides[1:]
        a_slices = np.lib.stride_tricks.as_strided(a, view_shape, strides)
        if mode == 'maxpool':
            a = np.max(a_slices, axis=(3, 4))
        elif mode == 'avgpool':
            a = np.mean(a_slices, axis=(3, 4))
        return a

    @staticmethod
    def __pool_backward(dout, a, stride, pad, mode):
        if mode == 'maxpool':
            a = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))
            view_shape = (a.shape[0], (a.shape[1] - stride[0]) // stride[0] + 1 \
                              , (a.shape[2] - stride[1]) // stride[1] + 1, stride[0], stride[1], a.shape[3])
            strides = (a.strides[0], stride[0] * a.strides[1], stride[1] * a.strides[2]) + a.strides[1:]
            a_slices = np.lib.stride_tricks.as_strided(a, view_shape, strides)

            # Convert slices to mask
            s = a_slices.shape
            a_slices = np.reshape(a_slices,
                                  (a_slices.shape[0], a_slices.shape[1], a_slices.shape[2], -1, a_slices.shape[5]))
            idx = a_slices.argmax(axis=3, keepdims=True)
            a_slices = np.zeros(a_slices.shape)
            np.put_along_axis(a_slices, idx, 1, axis=3)
            a_slices = np.reshape(a_slices, s)

            dout = dout[:, :, :, None, None, :]
            dout = dout * a_slices

            dout = np.reshape(np.transpose(dout, (0, 1, 3, 2, 4, 5)), (a.shape[0], a.shape[1] - (a.shape[1] % stride[0]) \
                                                                           , a.shape[2] - (a.shape[2] % stride[1]),
                                                                       a.shape[3]))
            d = np.zeros(a.shape)
            d[:, :dout.shape[1], :dout.shape[2], :] = dout
            dout = d

        elif mode == 'avgpool':
            pad_bottom = (a.shape[1] + 2 * pad - stride[0]) % stride[0]
            pad_right = (a.shape[2] + 2 * pad - stride[1]) % stride[1]
            for r in range(stride[0] - 1):
                dout = np.insert(dout, range(1, dout.shape[1], r + 1), 0, axis=1)
            for r in range(stride[1] - 1):
                dout = np.insert(dout, range(1, dout.shape[2], r + 1), 0, axis=2)
            for _ in range(pad_bottom):
                dout = np.insert(dout, dout.shape[1], 0, axis=1)
            for _ in range(pad_right):
                dout = np.insert(dout, dout.shape[2], 0, axis=2)
            dout = np.pad(dout, ((0, 0), (stride[0] - 1, stride[0] - 1) \
                                     , (stride[1] - 1, stride[1] - 1), (0, 0)), 'constant', constant_values=(0, 0))

            strides = (
            dout.strides[0], dout.strides[1], dout.strides[2], dout.strides[1], dout.strides[2], dout.strides[3])
            dout = np.lib.stride_tricks.as_strided(dout,
                                                   shape=(dout.shape[0], a.shape[1] + 2 * pad, a.shape[2] + 2 * pad \
                                                              , stride[0], stride[1], dout.shape[3]), strides=strides,
                                                   writeable=False)
            dout = np.mean(dout, axis=(3, 4))

        if pad > 0:
            dout = dout[:, pad:-pad, pad:-pad, :]
        return dout

    @staticmethod
    def __conv_forward(a, w, b, stride, pad, batch_norm):
        a = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))
        view_shape = (
                     a.shape[0], (a.shape[1] - w.shape[0]) // stride[0] + 1, (a.shape[2] - w.shape[1]) // stride[1] + 1) \
                     + w.shape[:-1]
        strides = (a.strides[0], stride[0] * a.strides[1], stride[1] * a.strides[2]) + a.strides[1:]
        a_slices = np.lib.stride_tricks.as_strided(a, view_shape, strides)
        if batch_norm:
            output = np.tensordot(a_slices, w, axes=[(3, 4, 5), (0, 1, 2)])
        else:
            output = np.tensordot(a_slices, w, axes=[(3, 4, 5), (0, 1, 2)]) + b
        return output

    @staticmethod
    def __conv_backward(dout, a, w, b, stride, pad, batch_norm):

        if not batch_norm:
            db = np.sum(dout, axis=(0, 1, 2), keepdims=True)
        a_pad = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))
        view_shape = (a_pad.shape[0], (a_pad.shape[1] - w.shape[0]) // stride[0] + 1 \
                          , (a_pad.shape[2] - w.shape[1]) // stride[1] + 1) + w.shape[:-1]
        strides = (a_pad.strides[0], stride[0] * a_pad.strides[1], stride[1] * a_pad.strides[2]) + a_pad.strides[1:]
        a_slices = np.lib.stride_tricks.as_strided(a_pad, view_shape, strides)
        dW = np.tensordot(a_slices, dout, axes=[(0, 1, 2), (0, 1, 2)])

        Wrot180 = np.rot90(w, 2, axes=(0, 1))
        pad_bottom = (a.shape[1] + 2 * pad - w.shape[0]) % stride[0]
        pad_right = (a.shape[2] + 2 * pad - w.shape[1]) % stride[1]
        for r in range(stride[0] - 1):
            dout = np.insert(dout, range(1, dout.shape[1], r + 1), 0, axis=1)
        for r in range(stride[1] - 1):
            dout = np.insert(dout, range(1, dout.shape[2], r + 1), 0, axis=2)
        for _ in range(pad_bottom):
            dout = np.insert(dout, dout.shape[1], 0, axis=1)
        for _ in range(pad_right):
            dout = np.insert(dout, dout.shape[2], 0, axis=2)
        dout = np.pad(dout, ((0, 0), (w.shape[0] - 1, w.shape[0] - 1), (w.shape[1] - 1, w.shape[1] - 1), (0, 0)) \
                      , 'constant', constant_values=(0, 0))

        strides = (dout.strides[0], dout.strides[1], dout.strides[2], dout.strides[1], dout.strides[2], dout.strides[3])
        dout = np.lib.stride_tricks.as_strided(dout, shape=(dout.shape[0], a.shape[1] + 2 * pad, a.shape[2] + 2 * pad \
                                                                , w.shape[0], w.shape[1], dout.shape[3]),
                                               strides=strides, writeable=False)
        dout = np.tensordot(dout, Wrot180, axes=([3, 4, 5], [0, 1, 3]))
        if pad > 0:
            dout = dout[:, pad:-pad, pad:-pad, :]
        if batch_norm:
            return dout, dW
        else:
            return dout, dW, db

    def __forward_prop(self, update=True):
        self.__layer_dropouts = {}
        self.__layer_activations = {}
        self.__bn_values = {}
        self.__layer_activations[f"A{0}"] = self.__X_minibatch
        for l in range(1, self.__L + 1):
            if self.__layers[l - 1] == "conv":
                Z = self.__conv_forward(self.__layer_activations[f"A{l - 1}"], self.__parameters[f"W{l}"] \
                                        , self.__parameters[f"b{l}"], self.__layer_dims[l][1] \
                                        , self.__layer_dims[l][2], self.__batch_norm[l - 1])
                if self.__batch_norm[l - 1]:
                    batch_mean = np.mean(Z, axis=0, keepdims=True)
                    self.__bn_values[f"var_{l}"] = np.var(Z, axis=0, keepdims=True)
                    self.__bn_values[f"Znorm_{l}"] = (Z - batch_mean) / np.sqrt(
                        self.__bn_values[f"var_{l}"] + self.__epsilon)
                    Z = self.__bn_values[f"Znorm_{l}"] * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]

                    if update:
                        self.__batch_norm_params[f"V_mean{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_mean{l}"] \
                                                                 + (1 - self.__bn_momentum) * batch_mean
                        self.__batch_norm_params[f"V_var{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_var{l}"] \
                                                                + (1 - self.__bn_momentum) * self.__bn_values[
                                                                    f"var_{l}"]
            elif self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                Z = self.__pool_forward(self.__layer_activations[f"A{l - 1}"], self.__layer_dims[l][0] \
                                        , self.__layer_dims[l][1], self.__layers[l - 1])
                if self.__batch_norm[l - 1]:
                    batch_mean = np.mean(Z, axis=0, keepdims=True)
                    self.__bn_values[f"var_{l}"] = np.var(Z, axis=0, keepdims=True)
                    self.__bn_values[f"Znorm_{l}"] = (Z - batch_mean) / np.sqrt(
                        self.__bn_values[f"var_{l}"] + self.__epsilon)
                    Z = self.__bn_values[f"Znorm_{l}"] * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]

                    if update:
                        self.__batch_norm_params[f"V_mean{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_mean{l}"] \
                                                                 + (1 - self.__bn_momentum) * batch_mean
                        self.__batch_norm_params[f"V_var{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_var{l}"] \
                                                                + (1 - self.__bn_momentum) * self.__bn_values[
                                                                    f"var_{l}"]
            else:
                if self.__batch_norm[l - 1]:
                    Z = np.dot(self.__layer_activations[f"A{l - 1}"], self.__parameters[f"W{l}"])
                    batch_mean = np.mean(Z, axis=0, keepdims=True)
                    self.__bn_values[f"var_{l}"] = np.var(Z, axis=0, keepdims=True)
                    self.__bn_values[f"Znorm_{l}"] = (Z - batch_mean) / np.sqrt(
                        self.__bn_values[f"var_{l}"] + self.__epsilon)
                    Z = self.__bn_values[f"Znorm_{l}"] * self.__parameters[f"gamma{l}"] + self.__parameters[f"b{l}"]

                    if update:
                        self.__batch_norm_params[f"V_mean{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_mean{l}"] \
                                                                 + (1 - self.__bn_momentum) * batch_mean
                        self.__batch_norm_params[f"V_var{l}"] = self.__bn_momentum * self.__batch_norm_params[
                            f"V_var{l}"] \
                                                                + (1 - self.__bn_momentum) * self.__bn_values[
                                                                    f"var_{l}"]

                else:
                    Z = np.dot(self.__layer_activations[f"A{l - 1}"], self.__parameters[f"W{l}"]) + self.__parameters[
                        f"b{l}"]

            if self.__layers[l - 1] == "conv" or self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                if self.__layers[l] == "dense":
                    Z = np.reshape(Z, (Z.shape[0], -1))

            A = self.__forward_activations[l - 1](Z)
            self.__layer_dropouts[f"D{l}"] = (np.random.rand(*A.shape) <= self.__keep_prob[l - 1]).astype(int)
            self.__layer_activations[f"A{l}"] = A * self.__layer_dropouts[f"D{l}"] / self.__keep_prob[l - 1]

    def __backward_prop(self):
        self.__grads = {}
        dZ = (self.__layer_activations[f"A{self.__L}"] - self.__Y_minibatch)

        for l in reversed(range(1, self.__L + 1)):
            if l != self.__L:
                dZ = dA * self.__backward_activations[l - 1](self.__layer_activations[f"A{l}"]) \
                     * self.__layer_dropouts[f"D{l}"] / self.__keep_prob[l - 1]

            if self.__layers[l - 1] == "conv":
                if self.__layers[l] == "dense":
                    dZ = np.reshape(dZ, ((dZ.shape[0],) + self.__outputs[l]))

                if self.__batch_norm[l - 1]:
                    self.__grads[f"dgamma{l}"] = np.mean(dZ * self.__bn_values[f"Znorm_{l}"], axis=0, keepdims=True)
                    db = np.mean(dZ, axis=0, keepdims=True)
                    dZnorm = dZ * self.__parameters[f"gamma{l}"]
                    dZ = (self.__curr_batch_size * dZnorm - np.sum(dZnorm, axis=0, keepdims=True) \
                          - self.__bn_values[f"Znorm_{l}"] * np.sum(dZnorm * self.__bn_values[f"Znorm_{l}"], \
                                                                    axis=0, keepdims=True)) \
                         / (self.__curr_batch_size * np.sqrt(self.__bn_values[f"var_{l}"] + self.__epsilon))
                    dA, dW = self.__conv_backward(dZ, self.__layer_activations[f"A{l - 1}"], self.__parameters[f"W{l}"] \
                                                  , self.__parameters[f"b{l}"], self.__layer_dims[l][1],
                                                  self.__layer_dims[l][2] \
                                                  , self.__batch_norm[l - 1])
                else:
                    dA, dW, db = self.__conv_backward(dZ, self.__layer_activations[f"A{l - 1}"],
                                                      self.__parameters[f"W{l}"] \
                                                      , self.__parameters[f"b{l}"], self.__layer_dims[l][1],
                                                      self.__layer_dims[l][2] \
                                                      , self.__batch_norm[l - 1])
                    db = db / self.__curr_batch_size

                dW = (dW + self.__lambd * self.__parameters[f"W{l}"]) / self.__curr_batch_size
            elif self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                if self.__layers[l] == "dense":
                    dZ = np.reshape(dZ, ((dZ.shape[0],) + self.__outputs[l]))
                if self.__batch_norm[l - 1]:
                    self.__grads[f"db{l}"] = np.mean(dZ, axis=0, keepdims=True)
                    self.__grads[f"dgamma{l}"] = np.mean(dZ * self.__bn_values[f"Znorm_{l}"], axis=0, keepdims=True)
                    dZnorm = dZ * self.__parameters[f"gamma{l}"]
                    dZ = (self.__curr_batch_size * dZnorm - np.sum(dZnorm, axis=0, keepdims=True) \
                          - self.__bn_values[f"Znorm_{l}"] * np.sum(dZnorm * self.__bn_values[f"Znorm_{l}"], \
                                                                    axis=0, keepdims=True)) \
                         / (self.__curr_batch_size * np.sqrt(self.__bn_values[f"var_{l}"] + self.__epsilon))
                dA = self.__pool_backward(dZ, self.__layer_activations[f"A{l - 1}"], self.__layer_dims[l][0] \
                                          , self.__layer_dims[l][1], self.__layers[l - 1])
            else:
                db = np.mean(dZ, axis=0, keepdims=True)

                if self.__batch_norm[l - 1]:
                    self.__grads[f"dgamma{l}"] = np.mean(dZ * self.__bn_values[f"Znorm_{l}"], axis=0, keepdims=True)
                    dZnorm = dZ * self.__parameters[f"gamma{l}"]
                    dZ = (self.__curr_batch_size * dZnorm - np.sum(dZnorm, axis=0, keepdims=True) \
                          - self.__bn_values[f"Znorm_{l}"] * np.sum(dZnorm * self.__bn_values[f"Znorm_{l}"], \
                                                                    axis=0, keepdims=True)) \
                         / (self.__curr_batch_size * np.sqrt(self.__bn_values[f"var_{l}"] + self.__epsilon))

                dW = np.dot(self.__layer_activations[f"A{l - 1}"].T, dZ) / self.__curr_batch_size + \
                     self.__lambd * self.__parameters[f"W{l}"] / self.__curr_batch_size
                dA = np.dot(dZ, self.__parameters[f"W{l}"].T)
            if self.__layers[l - 1] == "conv" or self.__layers[l - 1] == "dense":
                self.__grads[f"dW{l}"], self.__grads[f"db{l}"] = dW, db

    def __update_parameters(self):
        for l in range(1, self.__L + 1):

            if self.__layers[l - 1] == "maxpool" or self.__layers[l - 1] == "avgpool":
                if self.__batch_norm[l - 1]:
                    self.__adam_parameters[f"V_dgamma{l}"] = self.__beta1 * self.__adam_parameters[f"V_dgamma{l}"] \
                                                             + (1 - self.__beta1) * self.__grads[f"dgamma{l}"]
                    self.__adam_parameters[f"S_dgamma{l}"] = self.__beta2 * self.__adam_parameters[f"S_dgamma{l}"] \
                                                             + (1 - self.__beta2) * self.__grads[f"dgamma{l}"] ** 2
                    V_dgamma_corr = self.__adam_parameters[f"V_dgamma{l}"] / (1 - self.__beta1 ** \
                                                                              (
                                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))
                    S_dgamma_corr = self.__adam_parameters[f"S_dgamma{l}"] / (1 - self.__beta2 ** \
                                                                              (
                                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))
                    dgamma_final = V_dgamma_corr / (np.sqrt(S_dgamma_corr) + self.__epsilon)
                    self.__parameters[f"gamma{l}"] = self.__parameters[f"gamma{l}"] - self.__alpha * dgamma_final

                    self.__adam_parameters[f"V_db{l}"] = self.__beta1 * self.__adam_parameters[f"V_db{l}"] \
                                                         + (1 - self.__beta1) * self.__grads[f"db{l}"]
                    self.__adam_parameters[f"S_db{l}"] = self.__beta2 * self.__adam_parameters[f"S_db{l}"] \
                                                         + (1 - self.__beta2) * self.__grads[f"db{l}"] ** 2
                    V_db_corr = self.__adam_parameters[f"V_db{l}"] / (1 - self.__beta1 ** \
                                                                      (
                                                                                  self.__iteration * self.__batch_size + self.__batch_num + 1))
                    S_db_corr = self.__adam_parameters[f"S_db{l}"] / (1 - self.__beta2 ** \
                                                                      (
                                                                                  self.__iteration * self.__batch_size + self.__batch_num + 1))
                    db_final = V_db_corr / (np.sqrt(S_db_corr) + self.__epsilon)
                    self.__parameters[f"b{l}"] = self.__parameters[f"b{l}"] - self.__alpha * db_final
                continue

            self.__adam_parameters[f"V_dW{l}"] = self.__beta1 * self.__adam_parameters[f"V_dW{l}"] \
                                                 + (1 - self.__beta1) * self.__grads[f"dW{l}"]
            self.__adam_parameters[f"V_db{l}"] = self.__beta1 * self.__adam_parameters[f"V_db{l}"] \
                                                 + (1 - self.__beta1) * self.__grads[f"db{l}"]
            self.__adam_parameters[f"S_dW{l}"] = self.__beta2 * self.__adam_parameters[f"S_dW{l}"] \
                                                 + (1 - self.__beta2) * self.__grads[f"dW{l}"] ** 2
            self.__adam_parameters[f"S_db{l}"] = self.__beta2 * self.__adam_parameters[f"S_db{l}"] \
                                                 + (1 - self.__beta2) * self.__grads[f"db{l}"] ** 2

            V_dW_corr = self.__adam_parameters[f"V_dW{l}"] / (1 - self.__beta1 ** \
                                                              (
                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))
            V_db_corr = self.__adam_parameters[f"V_db{l}"] / (1 - self.__beta1 ** \
                                                              (
                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))
            S_dW_corr = self.__adam_parameters[f"S_dW{l}"] / (1 - self.__beta2 ** \
                                                              (
                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))
            S_db_corr = self.__adam_parameters[f"S_db{l}"] / (1 - self.__beta2 ** \
                                                              (
                                                                          self.__iteration * self.__batch_size + self.__batch_num + 1))

            dW_final = V_dW_corr / (np.sqrt(S_dW_corr) + self.__epsilon)
            db_final = V_db_corr / (np.sqrt(S_db_corr) + self.__epsilon)

            self.__parameters[f"W{l}"] = self.__parameters[f"W{l}"] - self.__alpha * dW_final
            self.__parameters[f"b{l}"] = self.__parameters[f"b{l}"] - self.__alpha * db_final

            if self.__batch_norm[l - 1]:
                self.__adam_parameters[f"V_dgamma{l}"] = self.__beta1 * self.__adam_parameters[f"V_dgamma{l}"] \
                                                         + (1 - self.__beta1) * self.__grads[f"dgamma{l}"]
                self.__adam_parameters[f"S_dgamma{l}"] = self.__beta2 * self.__adam_parameters[f"S_dgamma{l}"] \
                                                         + (1 - self.__beta2) * self.__grads[f"dgamma{l}"] ** 2
                V_dgamma_corr = self.__adam_parameters[f"V_dgamma{l}"] / (1 - self.__beta1 ** \
                                                                          (
                                                                                      self.__iteration * self.__batch_size + self.__batch_num + 1))
                S_dgamma_corr = self.__adam_parameters[f"S_dgamma{l}"] / (1 - self.__beta2 ** \
                                                                          (
                                                                                      self.__iteration * self.__batch_size + self.__batch_num + 1))
                dgamma_final = V_dgamma_corr / (np.sqrt(S_dgamma_corr) + self.__epsilon)
                self.__parameters[f"gamma{l}"] = self.__parameters[f"gamma{l}"] - self.__alpha * dgamma_final

    def __gradient_descent(self):

        start_iter = self.__iteration

        for self.__iteration in range(self.__iteration, self.__num_epochs):
            self.__alpha = self.__alpha0 / (1 + self.__decay_rate * (self.__iteration // self.__decay_interval))

            permutation = list(np.random.permutation(self.__m))
            self.__X_shuffled = self.__X[permutation, :]
            self.__Y_shuffled = self.__Y[permutation, :]

            cost = 0
            self.__curr_batch_size = self.__batch_size

            for self.__batch_num in range(self.__total_batches):
                if self.__batch_num != self.__total_batches - 1:
                    self.__X_minibatch = self.__X_shuffled[self.__batch_num * self.__batch_size: \
                                                           (self.__batch_num + 1) * self.__batch_size, :]
                    self.__Y_minibatch = self.__Y_shuffled[self.__batch_num * self.__batch_size: \
                                                           (self.__batch_num + 1) * self.__batch_size, :]
                else:
                    self.__X_minibatch = self.__X_shuffled[self.__batch_num * self.__batch_size:, :]
                    self.__Y_minibatch = self.__Y_shuffled[self.__batch_num * self.__batch_size:, :]
                    self.__curr_batch_size = self.__X_minibatch.shape[0]

                self.__forward_prop()
                cost += self.__cost_function()
                self.__backward_prop()
                self.__update_parameters()

            cost = cost / self.__total_batches
            self.__cost_history.append(cost)

            if (self.__iteration - start_iter) % (max((self.__num_epochs - start_iter) // 10, 1)) == 0 or \
                    self.__iteration == self.__num_epochs - 1:
                print(f"Cost for epoch {self.__iteration + 1} is {cost}")
                if self.__grad_check:
                    self.__forward_prop(update=False)
                    self.__backward_prop()
                    print(f"Output of gradient check: {self.__gradient_check(self.__cost_function)}")
        self.__iteration = self.__num_epochs

