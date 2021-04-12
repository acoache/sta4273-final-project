import numpy as np
import tensorflow as tf

"""
# Model classes

* EnsembleQNetwork -- Ensemble of neural networks
* ModelWithPrior -- Combination a model and a prior together
* EnsembleWithPrior -- Ensemble of neural networks with priors
* NoisyDense -- Dense layer to create a NoisyNet
* EnsembleQNoisyNet -- Ensemble of NoisyNets
* GaussianParamNet_2D -- Neural Net to parametrize full Gaussian distribution (through the mean and the Cholesky)
* Ensemble_GaussianParamNet_2D -- Ensemble of GaussianParamNet_2D
"""

class EnsembleQNetwork(tf.keras.Model):
    # constructor
    def __init__(self, input_size, hidden_sizes, output_size, num_ensemble):
        super(EnsembleQNetwork, self).__init__()
        self.num_ensemble = num_ensemble
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        # for each ensemble
        for idx in range(self.num_ensemble):
            # input layer
            input = tf.keras.layers.InputLayer(input_shape = (input_size,))
            self.input_layer.append(input)

            # hidden layer(s)
            hidden_ens = []
            for hidden_size in hidden_sizes:
                hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
                hidden_ens.append(hidden)
            self.hidden_layers.append(hidden_ens)

            # output layer
            output = tf.keras.layers.Dense(output_size, activation='linear')
            self.output_layer.append(output)

    @tf.function
    def call(self, inputs, **kwargs):
        output = []
        
        # for each ensemble
        for idx in range(self.num_ensemble):
            x = self.input_layer[idx](inputs)

            for hidden in self.hidden_layers[idx]:
                x = hidden(x)

            x = self.output_layer[idx](x)
            output.append(x)

        return tf.stack(output, axis=1)

class ModelWithPrior(tf.keras.Model):
    # Constructor
    def __init__(self, model, prior, prior_scale):
        super(ModelWithPrior, self).__init__()
        self._model_network = model
        self._prior_network = prior
        self.prior_scale = prior_scale
  
    @tf.function
    def call(self, inputs, **kwargs):
        # Prior model not optimized
        prior_output = tf.stop_gradient(self._prior_network(inputs))
        model_output = self._model_network(inputs)

        return model_output + self.prior_scale * prior_output

class EnsembleWithPrior(tf.keras.Model):
    # Constructor
    def __init__(self, input_size, hidden_sizes, output_size, num_ensemble, prior_scale):
        super(EnsembleWithPrior, self).__init__()
        self.num_ensemble = num_ensemble
        self.model = EnsembleQNetwork(input_size, hidden_sizes, output_size, num_ensemble)
        self.prior = EnsembleQNetwork(input_size, hidden_sizes, output_size, num_ensemble)
        self.network = ModelWithPrior(self.model, self.prior, prior_scale)

    @tf.function
    def call(self, inputs, **kwargs):
        return self.network(inputs, **kwargs)

class NoisyDense(tf.keras.layers.Layer):
    # constructor
    def __init__(self, input_size, output_size, mu_init=1, sigma_init=0.5, activation=None):
        super(NoisyDense, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)

        self.reset_parameters()

    # create trainable variables (mu / sigma) for weights & biases
    def reset_parameters(self):
        mu_range = self.mu_init / np.sqrt(self.input_size)
        sigma_value = self.sigma_init / np.sqrt(self.input_size)
        
        # initializers, see section 3.2 of the original paper
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(sigma_value)
        
        # create variables
        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(self.input_size, self.output_size),
                                                                  dtype='float32'),
                                      trainable=True)
        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.input_size, self.output_size),
                                                                                dtype='float32'),
                                        trainable=True)
        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(self.output_size,),
                                                                dtype='float32'),
                                    trainable=True)
        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.output_size,),
                                                                            dtype='float32'),
                                      trainable=True)
  
    # factorised Gaussian noise, see section 3 b) of the original paper on NoisyNets
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_size)
        epsilon_out = self._scale_noise(self.output_size)
        self.weight_epsilon = tf.tensordot(epsilon_in, epsilon_out, axes=0)
        self.bias_epsilon = epsilon_out
  
    # Epsilon, see equations (10)-(11) in the original paper on NoisyNets
    def _scale_noise(self, size):
        # zero-mean noise
        x = tf.random.normal([size])
        return tf.sign(x) * tf.sqrt(tf.abs(x))
  
    def call(self, x):
        # sample noise
        self.reset_noise()
        
        # noisy weights & biases
        self.weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        self.bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        # noisy layer
        output = tf.matmul(x, self.weight) + self.bias
        
        # activation function if needed
        if self.activation is not None:
            output = self.activation(output)

        return output

class EnsembleQNoisyNet(tf.keras.Model):
    # constructor
    def __init__(self, input_size, hidden_sizes, output_size, num_ensemble, sigma_init=0.5):
        super(EnsembleQNoisyNet, self).__init__()
        self.num_ensemble = num_ensemble
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

        # for each ensemble
        for idx in range(self.num_ensemble):
            # input layer
            input = tf.keras.layers.InputLayer(input_shape = (input_size,))
            self.input_layer.append(input)
            
            # noisy hidden layer(s)
            hidden_ens = []
            hidden = NoisyDense(input_size=input_size, output_size=hidden_sizes[0], 
                              mu_init=1, sigma_init=sigma_init, activation='relu')
            hidden_ens.append(hidden)
            for h_idx in range(len(hidden_sizes)-1):
                hidden = NoisyDense(input_size=hidden_sizes[h_idx], output_size=hidden_sizes[h_idx+1], 
                                  mu_init=1, sigma_init=sigma_init, activation='relu')
                hidden_ens.append(hidden)
            self.hidden_layers.append(hidden_ens)
            
            # noisy output layer
            output = NoisyDense(input_size=hidden_sizes[-1], output_size=output_size, 
                              mu_init=1, sigma_init=sigma_init, activation='linear')
            self.output_layer.append(output)

    @tf.function
    def call(self, inputs, **kwargs):
        output = []

        for idx in range(self.num_ensemble):
            x = self.input_layer[idx](inputs)
            
            for hidden in self.hidden_layers[idx]:
                x = hidden(x)
                
            x = self.output_layer[idx](x)
            output.append(x)

        return tf.stack(output, axis=1)

# currently only works for 2D, output_size not used
class GaussianParamNet_2D(tf.keras.Model):
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GaussianParamNet_2D, self).__init__()
        
        # mu corresponds to mean
        # diag corresponds to diagonal of Cholesky
        # odiag corresponds to lower off-diagonal of Cholesky (in 2D case, just a scalar)
        
        # input layer
        self.mu_input_layer = tf.keras.layers.InputLayer(input_shape = (input_size,))
        self.diag_input_layer = tf.keras.layers.InputLayer(input_shape = (input_size,))
        self.odiag_input_layer = tf.keras.layers.InputLayer(input_shape = (input_size,))

        # hidden layer(s)
        hidden_ens = []
        for hidden_size in hidden_sizes:
            hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
            hidden_ens.append(hidden)
        self.mu_hidden_layers = hidden_ens
        
        hidden_ens = []
        for hidden_size in hidden_sizes:
            hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
            hidden_ens.append(hidden)
        self.diag_hidden_layers = hidden_ens
        
        hidden_ens = []
        for hidden_size in hidden_sizes:
            hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
            hidden_ens.append(hidden)
        self.odiag_hidden_layers = hidden_ens

        # output layer (hardcoded to be [L[0,0], L[1,1], L[1,0]])
        self.mu_output_layer = tf.keras.layers.Dense(2, activation='linear')
        self.diag_output_layer = tf.keras.layers.Dense(2, activation='linear')
        self.odiag_output_layer = tf.keras.layers.Dense(1, activation='linear')

    @tf.function
    def call(self, inputs, **kwargs):
        
        # compute mean
        mu = self.mu_input_layer(inputs)

        for mu_hidden in self.mu_hidden_layers:
            mu = mu_hidden(mu)
        
        mu = self.mu_output_layer(mu)

        # compute diagonal
        diag = self.diag_input_layer(inputs)

        for diag_hidden in self.diag_hidden_layers:
            diag = diag_hidden(diag)
        
        diag = tf.math.exp(self.diag_output_layer(diag)) # diagonal should be nonneg
        
        # compute off-diagonal
        odiag = self.odiag_input_layer(inputs)

        for odiag_hidden in self.odiag_hidden_layers:
            odiag = odiag_hidden(odiag)
        
        odiag = self.odiag_output_layer(odiag)
    
        return mu, diag, odiag

# currently only works for 2D, output_size not used
class Ensemble_GaussianParamNet_2D(tf.keras.Model):
    # constructor
    def __init__(self, input_size, hidden_sizes, output_size, num_ensemble):
        super(Ensemble_GaussianParamNet_2D, self).__init__()
        self.num_ensemble = num_ensemble
        self.networks = []

        # for each ensemble
        for _ in range(self.num_ensemble):
            self.networks.append(GaussianParamNet_2D(input_size, hidden_sizes, output_size))

    @tf.function
    def call(self, inputs, **kwargs):
        mu_output = []
        diag_output = []
        odiag_output = []

        for idx in range(self.num_ensemble):
            mu_tmp, diag_tmp, odiag_tmp = self.networks[idx](inputs, **kwargs)
            mu_output.append(mu_tmp)
            diag_output.append(diag_tmp)
            odiag_output.append(odiag_tmp)
    
        return tf.stack(mu_output, axis=1), tf.stack(diag_output, axis=1), tf.stack(odiag_output, axis=1)

