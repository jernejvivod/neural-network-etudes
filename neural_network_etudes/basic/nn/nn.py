import numpy as np

import neural_network_etudes.basic.aux.funcs as funcs


class Network:

    def __init__(self, config):
        """
        weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        the weights for that layer of dimensions size(L+1) X size(L)
        the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        in layer L is therefore size(L+1).
        The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        """

        # Store training parameters.
        self._sizes = config.sizes
        self._mini_batch_size = config.mini_batch_size
        self._eta = config.eta
        self._epochs = config.epochs
        self._dropout = config.dropout
        if self._dropout:
            self._p_dropout = config.p_dropout
            self._dropout_masks = None
            if len(config.dropout_l) == 0:
                self._dropout_l = [True for _ in range(len(self.weights) + 1)]
            else:
                self._dropout_l = config.dropout_l
        self._l2_reg = config.l2_reg
        if self._l2_reg:
            self._lbd = config.lbd
        self._adam = config.adam
        self._adaptive_lr = config.adaptive_lr
        if self._adaptive_lr:
            self._decay_rt = 0.3
            self._eta0 = config.eta

        if self._adam:
            self._adam_v_s = None
            self._beta_1 = config.beta_1
            self._beta_2 = config.beta_2

        self.weights = None
        self.biases = None

    def train(self, training_data, training_class):
        """
        Train neural network.

        Args:
            training_data (numpy.ndarray): training data array
            training_class (numpy.ndarray): training data class values array (ideal NN output)
        Returns:
            (list): loss for each epoch
        """

        # Initialize weights. Remember: w_{i, j} is a weight connecting j-th node in the previous layer with the i-th node
        # in the first layer -> the matrices are R^{sizes[i] \times sizes[i-1]}.
        self._sizes.insert(0, training_data.shape[0])
        self._sizes.append(training_class.shape[0])
        self.weights = [((2 / self._sizes[i - 1]) ** 0.5) * np.random.randn(self._sizes[i], self._sizes[i - 1]) for i in range(1, len(self._sizes))]

        # Initialize biases with zeros (R^{sizes[i+1] \times 1} for each node with inbound connections)
        self.biases = [np.zeros((x, 1)) for x in self._sizes[1:]]

        # If using Adam, store variables for the moving average computations for each layer.
        if self._adam:
            self._adam_v_s = [{'v_gw': np.zeros(self.weights[idx].shape, dtype=float),
                               'v_gb': np.zeros(self.biases[idx].shape, dtype=float),
                               's_gw': np.zeros(self.weights[idx].shape, dtype=float),
                               's_gb': np.zeros(self.biases[idx].shape, dtype=float)} for idx in range(len(self.weights))]

        # NOTE: the training data matrix is the transpose of the usual representation.

        # Get number of training examples.
        n = training_data.shape[1]

        # Initialize list for storing epoch losses.
        epoch_losses = list()

        # Go over epochs.
        for epoch_idx in range(self._epochs):

            print("Epoch {0}".format(epoch_idx))

            # If using adaptive learning rate, compute eta for next iteration.
            if self._adaptive_lr:
                self._eta = (1.0 / (1 + self._decay_rt * (epoch_idx + 1))) * self._eta0
                # print("eta={0}".format(eta))

            # Average loss.
            loss_acc = 0.0

            # Construct mini-batches.
            mini_batches = [
                (training_data[:, k:k + self._mini_batch_size], training_class[:, k:k + self._mini_batch_size]) for k in range(0, n, self._mini_batch_size)
            ]

            # Go over mini-batches.
            for it_idx, mini_batch in enumerate(mini_batches):

                # Compute forward pass to get output of neural network, the cached z-values and the cached a-values.
                output, z_s, a_s = self.forward_pass(mini_batch[0])

                # Run backpropagation to compute gradients of weights and biases.
                gw, gb = self.backward_pass(output, mini_batch[1], z_s, a_s)

                # Update neural network.
                if self._adam:
                    self.update_network_adam(gw, gb, self._eta, it_idx + 1)
                else:
                    self.update_network(gw, gb, self._eta)

                # Compute loss and add to average loss.
                loss = funcs.cross_entropy(mini_batch[1], output)
                loss_acc += loss

            # Compute average loss and append to list of losses.
            loss_avg = loss_acc / len(mini_batches)
            epoch_losses.append(loss_avg)

            # Print epoch information.
            print("Epoch {} complete".format(epoch_idx))
            print("Loss:" + str(loss_avg))

        # Return loss vs epoch.
        return epoch_losses

    def eval_network(self, validation_data, validation_class):
        """
        Evaluate input to network.

        Args:
            validation_data (numpy.ndarray): validation data array
            validation_class (numpy.ndarray): validation data class values array (ideal NN output)

        Returns:
            (float): computed classification accuracy
        """

        # Get number of examples in validation data.
        n = validation_data.shape[1]

        # Average loss
        loss_acc = 0.0

        # Initialize counter of true positives.
        tp = 0.0

        # Go over validation examples.
        for i in range(validation_data.shape[1]):
            # Get training example and transform to n_{0} \times m vector.
            example = np.expand_dims(validation_data[:, i], -1)

            # Get example ideal output.
            example_class = np.expand_dims(validation_class[:, i], -1)
            example_class_num = np.argmax(validation_class[:, i], axis=0)

            # Compute forward pass.
            output, z_s, activations = self.forward_pass(example)

            # Get output class (class with highest probability).
            output_num = np.argmax(output, axis=0)[0]

            # Check if classification correct (add 0 or 1 to counter of true positives).
            tp += int(example_class_num == output_num)

            # Compute cross-entropy loss of output.
            loss = funcs.cross_entropy(example_class, output)
            loss_acc += loss

        # Print loss and accuracy.
        vl = loss_acc / validation_data.shape[1]
        print("Validation Loss:" + str(vl))
        ca = tp / validation_data.shape[1]
        print("Classification accuracy: " + str(ca))

        # Return computed classification accuracy.
        return ca, vl

    def update_network(self, gw, gb, eta):
        """
        Update neural network using computed weights and biases gradients.

        Args:
            gw (list): List of weights gradients for each layer
            gb (list): List of biases gradients for each layer
            eta (float): the update scaling parameter
        """

        # Update weights for each layer.
        for idx in range(len(self.weights)):
            self.weights[idx] -= eta * gw[idx]

        # Update biases for each layer.
        for idx in range(len(self.biases)):
            self.biases[idx] -= eta * gb[idx]

    def update_network_adam(self, gw, gb, eta, it_idx):
        """
        Update neural network using computed weights and biases gradients using Adam optimization.

        Args:
            gw (list): List of weights gradients for each layer.
            gb (list): List of biases gradients for each layer.
            eta (float): Update scaling parameter.
            it_idx (float): Iteration index
        """

        # Update weights for each layer.
        for idx in range(len(self.weights)):
            self._adam_v_s[idx]['v_gw'] = self._beta_1 * self._adam_v_s[idx]['v_gw'] + (1 - self._beta_1) * gw[idx]
            self._adam_v_s[idx]['s_gw'] = self._beta_2 * self._adam_v_s[idx]['s_gw'] + (1 - self._beta_2) * (gw[idx] ** 2)
            v_gw_corr = self._adam_v_s[idx]['v_gw'] / (1 - self._beta_1 ** it_idx)
            s_gw_corr = self._adam_v_s[idx]['s_gw'] / (1 - self._beta_2 ** it_idx)
            self.weights[idx] -= eta * (v_gw_corr / (np.sqrt(s_gw_corr) + 1.0e-8))

        # Update biases for each layer.
        for idx in range(len(self.biases)):
            self._adam_v_s[idx]['v_gb'] = self._beta_1 * self._adam_v_s[idx]['v_gb'] + (1 - self._beta_1) * gb[idx]
            self._adam_v_s[idx]['s_gb'] = self._beta_2 * self._adam_v_s[idx]['s_gb'] + (1 - self._beta_2) * (gb[idx] ** 2)
            v_gb_corr = self._adam_v_s[idx]['v_gb'] / (1 - self._beta_1 ** it_idx)
            s_gb_corr = self._adam_v_s[idx]['s_gb'] / (1 - self._beta_2 ** it_idx)
            self.biases[idx] -= eta * (v_gb_corr / (np.sqrt(s_gb_corr) + 1.0e-8))

    def forward_pass(self, input_nn):
        """
        Compute the forward pass through the neural network.

        Args:
            input_nn (numpy.ndarray): Input data

        Returns:
            (tuple): output of the neural network, cached Z-values, cached activations.
        """

        # Initialize cache lists for z values and activations for each layer.
        z_cache = list()
        a_cache = [input_nn]  # The input to the neural network represents the initial activations.

        # Initialize dropout masks.
        if self._dropout:
            self._dropout_masks = [np.random.binomial(1, 1 - self._p_dropout, (self.weights[idx].shape[1], input_nn.shape[1]))
                                   for idx in range(len(self.weights))]

        # First activation is the input to the neural network.
        a = input_nn

        # If using dropout, apply dropout mask.
        if self._dropout and self._dropout_l[0]:
            a *= self._dropout_masks[0]
            a /= (1 - self._p_dropout)

        # Go over layers.
        for idx in range(len(self.weights) - 1):

            # Compute z values and activations for layer.
            z = self.weights[idx].dot(a) + self.biases[idx]
            a = funcs.sigmoid(z)

            # If using dropout, apply dropout mask.
            if self._dropout and self._dropout_l[idx + 1]:
                a *= self._dropout_masks[idx + 1]
                a /= (1 - self._p_dropout)

            # Cache z values and activations for layer.
            z_cache.append(z)
            a_cache.append(a)

        # Compute z and activations for final (output) layer.
        z = self.weights[-1].dot(a) + self.biases[-1]
        a = funcs.softmax(z)

        # Cahce z values and activations for final layer.
        z_cache.append(z)
        a_cache.append(a)

        # Return the activations for the final layer and the cached z values and activations.
        return a, z_cache, a_cache

    def backward_pass(self, output, target, z_s, activations):
        """
        Compute the backward pass through the neural network.

        Args:
            output (numpy.ndarray): Output of the forward pass.
            target (numpy.ndarray): Target values (ideal output).
            z_s (numpy.ndarray): z values
            activations (numpy.ndarray): Activations from the forward pass.

        Returns:
            (tuple): Gradient of weights, gradient of activations.
        """

        # Initialize lists for the weight and bias gradients for each layer.
        dw_s = list()
        db_s = list()

        # Compute the derivative of the softmax function on the output.
        dz = funcs.softmax_dz(output, target)

        # Compute the final layer weights derivatives.
        if self._l2_reg:
            dw = (dz.dot(activations[-2].T) + self._lbd * self.weights[-1]) / output.shape[1]
        else:
            dw = dz.dot(activations[-2].T) / output.shape[1]

        # Compute the final layer bias derivatives.
        db = np.sum(dz, axis=1, keepdims=True) / output.shape[1]

        # Compute the derivatives of the activations for the previous layer.
        da_prev = self.weights[-1].T.dot(dz)

        # If performing dropout on current layer.
        if self._dropout and self._dropout_l[-1]:
            da_prev *= self._dropout_masks[-1]
            da_prev /= (1 - self._p_dropout)

        # Store the weights and biases gradients for the final layer.
        dw_s.append(dw)
        db_s.append(db)

        # Go over the remaining layers.
        for idx in range(len(self.weights) - 1, 0, -1):

            # Compute the derivative of z values.
            dz = da_prev * (funcs.sigmoid(z_s[idx - 1]) * (1.0 - funcs.sigmoid(z_s[idx - 1])))

            # Compute the derivatives of the weights.
            if self._l2_reg:
                dw = (dz.dot(activations[idx - 1].T) + self._lbd * self.weights[idx - 1]) / output.shape[1]
            else:
                dw = dz.dot(activations[idx - 1].T) / output.shape[1]

            # Compute the derivatives of the biases.
            db = np.sum(dz, axis=1, keepdims=True) / output.shape[1]

            # If not dealing with first weights, compute derivatives of activations for
            # the previous layer.
            if idx > 1:
                da_prev = self.weights[idx - 1].T.dot(dz)

                # If performing dropout on current layer.
                if self._dropout and self._dropout_l[idx - 1]:
                    da_prev *= self._dropout_masks[idx - 1]
                    da_prev /= (1 - self._p_dropout)

            # Sore the weights and biases gradients for the layer.
            dw_s.insert(0, dw)
            db_s.insert(0, db)

        # Return the list of weights and biases gradients for the layers.
        return dw_s, db_s
