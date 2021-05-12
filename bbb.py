import numpy as np
import tensorflow as tf

from bbb_utils import *


# class WeightPriorVirtual:
#     def __init__(self):
#         pass
#
#     def calculate_kl(self,
#                      var_par):
#         pass


class WeightPriorStudent:
    def __init__(self,):
        pass
        # super(WeightPriorStudent, self).__init__()

    def calculate_kl(self,
                     var_par):
        W_mu = var_par["W_mu"]
        W_var = var_par["W_var"]
        W_log_alpha = var_par["W_log_alpha"]
        w_dims = var_par["w_dims"]
        use_bias = var_par["use_bias"]
        if use_bias:
            b_mu = var_par["b_mu"]
            b_var = var_par["b_var"]
            b_log_alpha = var_par["b_log_alpha"]
            b_dims = var_par["b_dims"]

        # This is known in closed form.
        W_kl_a = - 0.5 * tf.log(W_var)
        W_kl_b = tf.sqrt(1e-8 + tf.exp(W_log_alpha)) * get_random(tuple(w_dims), avg=0., std=1.)
        W_kl_c = tf.multiply(W_mu, 1.0 + W_kl_b)
        W_kl_d = tf.pow(W_kl_c, 2.0)
        W_kl_e = (1.0 + 1e-8) * tf.log(1e-8 + W_kl_d) / 2.0
        W_kl = W_kl_a + W_kl_e
        kl = tf.reduce_sum(W_kl)
        if use_bias:
            b_kl_a = - 0.5 * tf.log(b_var)
            b_kl_b = tf.sqrt(1e-8 + tf.exp(b_log_alpha)) * get_random(tuple(b_dims), avg=0., std=1.)
            b_kl_c = tf.multiply(b_mu, 1.0 + b_kl_b)
            b_kl_d = tf.pow(b_kl_c, 2.0)
            b_kl_e = (1.0 + 1e-8) * tf.log(1e-8 + b_kl_d) / 2.0
            b_kl = b_kl_a + b_kl_e
            kl = kl + tf.reduce_sum(b_kl)
        return kl


class WeightPriorARD:
    def __init__(self,):
        pass
        # super(WeightPriorARD, self).__init__()

    def calculate_kl(self,
                     var_par):
        W_mu = var_par["W_mu"]
        W_log_alpha = var_par["W_log_alpha"]
        use_bias = var_par["use_bias"]
        if use_bias:
            b_mu = var_par["b_mu"]
            b_log_alpha = var_par["b_log_alpha"]

        # This is known in closed form.
        W_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(W_log_alpha))) * tf.ones_like(W_mu))
        kl = W_kl
        if use_bias:
            b_kl = tf.reduce_sum(tf.log(1.0 + (1.0 / 1e-8 + tf.exp(b_log_alpha))) * tf.ones_like(b_mu))
            kl = kl + b_kl
        return kl


class WeightPriorMOG:
    def __init__(self,
                 sigma_prior,  # Should be list, even if just 1 value.
                 mixture_weights,
                 calculation_type):  # ["MC", "Closed"]
        # super(WeightPriorMOG, self).__init__()
        ######################## Initialising prior Gaussian mixture ###########################
        # If there are no mixture weights, initialise at equal mixtures
        if mixture_weights is None:
            mixture_weights = list()
            for i in range(len(sigma_prior)):
                mixture_weights.append(1.0 / len(sigma_prior))

        if len(sigma_prior) != len(mixture_weights):
            raise ValueError('Invalid Gaussian Mixture defined. ')

        for i in mixture_weights:
            if i < 0.0:
                raise ValueError('Invalid Mixture Weights. ')

        # Georgios TODO: Saw both np and statistics mean() in versions. Am keeping commented the other.
        mixture_weights_norm = np.mean(mixture_weights)
        # mixture_weights_norm = statistics.mean(mixture_weights)
        mixture_weights = [m_w / mixture_weights_norm for m_w in mixture_weights]

        self._sigma_prior = sigma_prior
        self._mixture_weights = mixture_weights
        self._calculation_type = calculation_type

        if self._calculation_type not in ["MC", "Closed"]:
            raise ValueError("Invalid KL-div calculation type.")

        if (self._calculation_type == "Closed") and (len(self._mixture_weights) > 1):
            raise NotImplementedError("No approximate closed form calculation of KL-div implemented for MoGs.")

        # if self.hparams.prior_type == 'mixed':
        #     # Georgios: Am a bit unsure what the below does, but looks super cool.
        #     p_s_m = [m_w * np.square(s) for m_w, s in zip(self._mixture_weights, self._sigma_prior)]
        #     p_s_m = np.sqrt(np.sum(p_s_m))
        #
        #     if hparams.prior_target == 0.218:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 2) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 4) - 1.0)
        #     elif hparams.prior_target == 4.25:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 212.078) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 424.93) - 1.0)
        #     elif hparams.prior_target == 6.555:
        #         self.rho_max_init = math.log(math.exp(p_s_m / 3265.708106) - 1.0)
        #         self.rho_min_init = math.log(math.exp(p_s_m / 6507.637711) - 1.0)

    def calculate_kl(self,
                     var_par):
        if self._calculation_type == "MC":
            W_mu = var_par["W_mu"]
            W_sigma = var_par["W_sigma"]
            W_sample_list = var_par["W_sample_list"]
            use_bias = var_par["use_bias"]
            if use_bias:
                b_mu = var_par["b_mu"]
                b_sigma = var_par["b_sigma"]
                b_sample_list = var_par["b_sample_list"]

            # KL-div of gaussian mixtures is intractable and requires an approximation
            # Here, MC-sampling using the weights sampled in call().
            if len(W_sample_list) < 1:
                # We could conceivably perform a new MC-sampling of weights -- too expensive.
                raise NotImplementedError("We have not sampled here -- need to implement alternative approximation.")

            log_prior = 0.
            log_var_posterior = 0.

            for W in W_sample_list:
                log_prior += scale_mixture_prior_generalised(W, self._sigma_prior, self._mixture_weights)
                log_var_posterior += tf.reduce_sum(log_gaussian(W, W_mu, W_sigma))

            if use_bias:
                for b in b_sample_list:
                    log_prior += scale_mixture_prior_generalised(b, self._sigma_prior, self._mixture_weights)
                    log_var_posterior += tf.reduce_sum(log_gaussian(b, b_mu, b_sigma))

            kl = (log_var_posterior - log_prior) / len(W_sample_list)
        elif self._calculation_type == "Closed":
            W_mu = var_par["W_mu"]
            W_sigma = var_par["W_sigma"]
            use_bias = var_par["use_bias"]
            if use_bias:
                b_mu = var_par["b_mu"]
                b_sigma = var_par["b_sigma"]
            # kl = 0.5 * (self.alpha * (self.M.pow(2) + logS.exp()) - logS).sum()
            # This is known in closed form.
            # kl = - 0.5 + tf.log(sigma2) - tf.log(sigma1) + (tf.pow(sigma1, 2) + tf.pow(mu1 - mu2, 2)) / (2 * tf.pow(sigma2, 2))

            sigma_2 = self._sigma_prior[0] + 1e-8

            sigma_1 = W_sigma + tf.constant(1e-8, dtype=tf.float32)
            mu_1 = W_mu

            kl_W = - 0.5 + tf.log(sigma_2) - tf.log(sigma_1) + (tf.pow(sigma_1, 2) + tf.pow(mu_1, 2)) / (
                     2 * tf.pow(sigma_2, 2))

            kl = tf.reduce_sum(kl_W)

            if use_bias:
                sigma_1 = b_sigma + tf.constant(1e-8, dtype=tf.float32)
                mu_1 = b_mu

                kl_b = - 0.5 + tf.log(sigma_2) - tf.log(sigma_1) + (tf.pow(sigma_1, 2) + tf.pow(mu_1, 2)) / (
                         2 * tf.pow(sigma_2, 2))

                kl = kl + tf.reduce_sum(kl_b)
        else:
            raise ValueError("Invalid calculation type.")

        # else:
        #     psm = tf.Variable(4.25)
        #     if self.built:
        #         sigW = tf.debugging.check_numerics(
        #             tf.sqrt(tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0)) + 1e-7), 'NaN or inf')
        #         sigb = tf.debugging.check_numerics(softplus(self.b_rho), 'NaN or inf')
        #         klW = tf.log(psm / (sigW + 1e-6)) + (tf.pow(sigW, 2.0) + tf.pow(self.W_mu, 2.0)) / (
        #                 2 * tf.pow(psm, 2.0) + 1e-6) - 0.5
        #         klb = tf.log(psm / (sigb + 1e-6)) + (tf.pow(sigb, 2.0) + tf.pow(self.b_mu, 2.0)) / (
        #                 2 * tf.pow(psm, 2.0) + 1e-6) - 0.5
        #         return tf.reduce_sum(klW) + tf.reduce_sum(klb)
        #     else:
        #         return tf.constant(0.0)

        return kl


class DenseReparameterisation(tf.keras.layers.Layer):
    def __init__(self, units, weight_prior,
                 variance_parameterisation_type,
                 use_clt,
                 activation=None,
                 uncertainty_propagation_type=None,
                 trainable=True,
                 use_bias=True,
                 name=None, reuse=None, **kwargs):
        super(DenseReparameterisation, self).__init__()
        self.units = units
        self.weight_prior = weight_prior
        self.variance_parameterisation_type = variance_parameterisation_type
        self.use_clt = use_clt
        self.activation = activation
        self.uncertainty_propagation_type = uncertainty_propagation_type
        self.trainable = trainable
        self.use_bias = use_bias
        self._name = name
        self.reuse = reuse

        #######################################################################################

        if variance_parameterisation_type not in ["additive",
                                                  "weight_wise",
                                                  "neuron_wise",
                                                  "layer_wise"]:
            raise NotImplementedError("Invalid variance parameterisation type.")

        self.rho_init = tf.random_uniform_initializer(-4.6, -3.9)

        self.built_q_network = False

        self.bayesian_loss = None

        # Variational parameters
        # Georgios: Keeping same names for posterior sharpening if possible.
        # self.W_mu = None
        # self.W_rho = None
        # if self.use_bias:
        #     self.b_mu = None
        #     self.b_rho = None

        # Georgios TODO: Am unsure if the below are used now. Could leave.
        # self.eps_out = tf.placeholder(tf.float32, (None, self.b_dims), name='eps_out')
        # self.eps_out_sample = get_random((self._batch_size, self.b_dims), avg=0., std=1.)

    def build(self, input_shape):

        if len(input_shape) > 2:
            raise ValueError("Invalid number of input tensor dimensions. Needs to be [batch_size, input_units].")

        self.w_dims = (int(input_shape[-1]), self.units)
        self.b_dims = (1, self.units)

        self.W_mu = self.add_weight(name=self._name + "_W_mu",
                                    shape=self.w_dims,
                                    dtype=tf.float32,
                                    initializer=tf.keras.initializers.glorot_normal(),
                                    regularizer=None,
                                    trainable=self.trainable,
                                    constraint=None,
                                    partitioner=None,
                                    use_resource=None)
        if self.variance_parameterisation_type == 'layer_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self._name + "_W_alpha",
                                               shape=(1, ),
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'neuron_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self._name + "_W_alpha",
                                               shape=self.b_dims,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'weight_wise':
            self.W_rho = None
            self.W_log_alpha = self.add_weight(name=self._name + "_W_alpha",
                                               shape=self.w_dims,  # self.w_dims,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(-4.0),
                                               regularizer=None,
                                               trainable=self.trainable,
                                               constraint=None,
                                               partitioner=None,
                                               use_resource=None)
        elif self.variance_parameterisation_type == 'additive':
            self.W_log_alpha = None
            self.W_rho = self.add_weight(name=self._name + "_W_rho",
                                         shape=self.w_dims,
                                         dtype=tf.float32,
                                         initializer=self.rho_init,
                                         regularizer=None,
                                         trainable=self.trainable,
                                         constraint=None,
                                         partitioner=None,
                                         use_resource=None)
        else:
            raise ValueError("Invalid variance parameterisation type.")
        if self.use_bias:
            self.b_mu = self.add_weight(name=self._name + "_b_mu",
                                        shape=self.b_dims,
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.01),
                                        regularizer=None,
                                        trainable=self.trainable,
                                        constraint=None,
                                        partitioner=None,
                                        use_resource=None)

            if self.variance_parameterisation_type == 'layer_wise':
                self.b_rho = None
                self.b_log_alpha = self.add_weight(name=self._name + "_b_alpha",
                                                   shape=(1, ),
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(-4.0),
                                                   regularizer=None,
                                                   trainable=self.trainable,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)
            elif self.variance_parameterisation_type in ['neuron_wise', 'weight_wise']:
                self.b_rho = None
                self.b_log_alpha = self.add_weight(name=self._name + "_b_alpha",
                                                   shape=self.b_dims,
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(-4.0),
                                                   regularizer=None,
                                                   trainable=self.trainable,
                                                   constraint=None,
                                                   partitioner=None,
                                                   use_resource=None)
            elif self.variance_parameterisation_type == 'additive':
                self.b_log_alpha = None
                self.b_rho = self.add_weight(name=self._name + "_b_rho",
                                             shape=self.b_dims,
                                             dtype=tf.float32,
                                             initializer=self.rho_init,
                                             regularizer=None,
                                             trainable=self.trainable,
                                             constraint=None,
                                             partitioner=None,
                                             use_resource=None)
            else:
                raise ValueError("Invalid variance parameterisation type.")

        # if self._use_posterior_sharpening:
        #     self.phi_W = tf.Variable(np.full(self.w_dims, 0.01), name=self._name + '_phi_W', shape=self.w_dims,
        #                                  dtype=tf.float32)
        #     self.phi_b = tf.Variable(np.full(self.b_dims, 0.01), name=self._name + '_phi_b', shape=self.b_dims,
        #                                  dtype=tf.float32)

    def call(self,  # Georgios The args were remade to somewhat resemble keras layers.
             inputs,
             training,
             n_samples,
             sample_type,  # "Sample", "MAP", "Variance"
             **kwargs):

        if n_samples < 1:
            raise ValueError("We assume at least 1 sample.")

        self.bayesian_loss = None

        input_tensor = inputs

        # The forward pass through one layer
        h_sum = 0.0
        h_list = []

        # The following are needed to calculate KL-div of variational posterior with prior.
        var_par = dict()
        var_par["use_bias"] = self.use_bias
        var_par["w_dims"] = self.w_dims
        var_par["b_dims"] = self.b_dims
        var_par["W_mu"] = None
        var_par["W_sigma"] = None
        var_par["W_var"] = None
        var_par["W_log_alpha"] = self.W_log_alpha
        var_par["W_rho"] = None
        var_par["b_mu"] = None
        var_par["b_sigma"] = None
        var_par["b_var"] = None
        var_par["b_log_alpha"] = self.b_log_alpha
        var_par["b_rho"] = None
        var_par["W_sample_list"] = []
        var_par["b_sample_list"] = []

        if self.variance_parameterisation_type == "layer_wise":
            out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var = self.layer_wise_variance(input_tensor)
        elif self.variance_parameterisation_type == "neuron_wise":
            out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var = self.layer_wise_variance(input_tensor)
        elif self.variance_parameterisation_type == "weight_wise":
            out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var = self.layer_wise_variance(input_tensor)
        elif self.variance_parameterisation_type == "additive":
            out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var = self.additive_variance(input_tensor)
        else:
            raise ValueError("Invalid variance parameterisation type.")

        out_mean = tf.where(tf.is_nan(out_mean), tf.zeros_like(out_mean), out_mean)
        out_std = tf.where(tf.is_nan(out_std), tf.zeros_like(out_std), out_std)

        if sample_type == "Sample":
            var_par["W_mu"] = W_mu
            var_par["W_var"] = W_var
            var_par["W_sigma"] = W_sigma
            if self.use_bias:
                var_par["b_mu"] = b_mu
                var_par["b_var"] = b_var
                var_par["b_sigma"] = b_sigma
        elif sample_type == "Variance":
            var_par["W_mu"] = tf.zeros_like(W_mu)
            var_par["W_var"] = W_var
            var_par["W_sigma"] = W_sigma
            if self.use_bias:
                var_par["b_mu"] = tf.zeros_like(b_mu)
                var_par["b_var"] = b_var
                var_par["b_sigma"] = b_sigma
        elif sample_type == "MAP":
            if n_samples > 1:
                raise ValueError("MAP is not sampling.")
            var_par["W_mu"] = W_mu
            var_par["W_var"] = tf.zeros_like(W_var)
            var_par["W_sigma"] = W_var
            if self.use_bias:
                var_par["b_mu"] = b_mu
                var_par["b_var"] = tf.zeros_like(b_var)
                var_par["b_sigma"] = b_var
        else:
            raise NotImplementedError("Invalid sampling type.")

        # Local Reparameterization
        if self.use_clt:
            for s in range(n_samples):
                out_mean_offset = tf.multiply(out_std, get_random(tf.shape(out_std), avg=0., std=1.))

                if sample_type == "Sample":
                    h_sample = out_mean + out_mean_offset
                    h_mean = out_mean
                    h_std = out_std
                elif sample_type == "Variance":
                    h_sample = out_mean_offset
                    h_mean = tf.zeros_like(out_mean)
                    h_std = out_std
                elif sample_type == "MAP":
                    if n_samples > 1:
                        raise ValueError("MAP is not sampling.")
                    h_sample = out_mean
                    h_mean = out_mean
                    h_std = tf.zeros_like(out_std)
                else:
                    raise NotImplementedError("Invalid sampling type.")
                h_sum += h_sample
                h_list.append(h_sample)

            if training:
                self.calculate_bayesian_loss(var_par=var_par)

        # Regular reparameterization.
        else:
            self.w_uncertainty = list()
            self.b_uncertainty = list()
            for s in range(n_samples):
                if sample_type == "Sample":
                    w_std = tf.multiply(var_par["W_sigma"], get_random(tuple(self.w_dims), avg=0., std=1.))
                    W_sample = var_par["W_mu"] + w_std

                    if self.use_bias:
                        b_std = tf.multiply(var_par["b_sigma"], get_random(tuple(self.b_dims), avg=0., std=1.))
                        b_sample = var_par["b_mu"] + b_std

                    self.w_uncertainty.append(w_std)
                    self.b_uncertainty.append(b_std)

                elif sample_type == "Variance":
                    W_sample = tf.multiply(var_par["W_sigma"],
                                           get_random(tuple(var_par["w_dims"]), avg=0., std=1.))
                    if self.use_bias:
                        b_sample = tf.multiply(var_par["b_sigma"],
                                               get_random((self.b_dims,), avg=0., std=1.))
                elif sample_type == "MAP":
                    if n_samples > 1:
                        raise ValueError("MAP is not sampling.")
                    W_sample = var_par["W_mu"]
                    if self.use_bias:
                        b_sample = var_par["b_mu"]
                else:
                    raise NotImplementedError("Invalid sampling type.")
                var_par["W_sample_list"].append(W_sample)
                if self.use_bias:
                    var_par["b_sample_list"].append(b_sample)

            for s in range(n_samples):
                W_sample = var_par["W_sample_list"][s]
                h_sample = tf.matmul(input_tensor, W_sample)
                h_sample = tf.where(tf.is_nan(h_sample), tf.zeros_like(h_sample), h_sample)
                h_sum += h_sample
                h_list.append(h_sample)

                b_sample = var_par["b_sample_list"][s]
                b_sample = tf.where(tf.is_nan(b_sample), tf.zeros_like(b_sample), b_sample)
                h_sum += b_sample
                h_list[s] += b_sample

            h_sum = tf.where(tf.is_nan(h_sum), tf.zeros_like(h_sum), h_sum)
            h_mean = h_sum / n_samples

            h_std = self.calc_h_std_from_samples(h_mean, h_list)
            # W_stacked = tf.stack(W_sample_list)
            # b_stacked = tf.stack(b_sample_list)
            # self.W = tf.reduce_sum(W_stacked, axis=0) / self._n_samples
            # self.b = tf.reduce_sum(b_stacked, axis=0) / self._n_samples

            if training:
                self.calculate_bayesian_loss(var_par=var_par)

        return h_mean, h_list, h_std

    def layer_wise_variance(self, x):
        W_mu = self.W_mu
        W_var = 1e-8 + tf.multiply(tf.exp(self.W_log_alpha),
                                   tf.pow(self.W_mu, 2.0))
        W_sigma = tf.sqrt(W_var)

        out_mean = tf.matmul(x, W_mu)
        out_std = tf.sqrt(1e-8 + tf.matmul(tf.pow(x, 2.0), W_var))

        if self.use_bias:
            b_mu = self.b_mu
            b_var = 1e-8 + tf.multiply(tf.exp(self.b_log_alpha),
                                       tf.pow(self.b_mu, 2.0))
            b_sigma = tf.sqrt(b_var)

            out_mean = out_mean + b_mu
            out_std = out_std + b_var
        else:
            b_mu = None
            b_var = None
            b_sigma = None
        return out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var

    def additive_variance(self, x):
        W_mu = self.W_mu
        W_sigma = softplus(self.W_rho)
        W_var = tf.pow(W_sigma, 2.0)

        out_mean = tf.matmul(x, W_mu)
        out_std = tf.sqrt(1e-8 + tf.matmul(tf.pow(x, 2.0), W_var))

        if self.use_bias:
            b_mu = self.b_mu
            b_sigma = softplus(self.b_rho)
            b_var = tf.pow(b_sigma, 2.0)

            out_mean = out_mean + b_mu
            out_std = out_std + b_var
        else:
            b_mu = None
            b_var = None
            b_sigma = None
        return out_mean, out_std, W_mu, W_sigma, W_var, b_mu, b_sigma, b_var

    def calc_h_std_from_samples(self, h_mean, h_list):
        if len(h_list) > 1:
            h_std = tf.pow(h_mean - h_list[0], 2.0)

            for s in range(1, len(h_list)):
                h_std = h_std + tf.pow(h_mean - h_list[s], 2.0)

            h_std = h_std / (len(h_list) - 1.0)
            h_std = tf.sqrt(1e-8 + h_std)
        else:
            h_std = tf.zeros_like(h_mean)

        return h_std

    def calculate_bayesian_loss(self,
                                var_par):

        kl = self.weight_prior.calculate_kl(var_par=var_par)

        self.kl = kl

    def get_bayesian_loss(self):
        return self.kl


# class DenseReparameterisation(DenseVariational):
#     def __init__(self, hparams, w_dims, b_dims, weight_prior,
#                  trainable=None, units=None,
#                  use_bias=True,
#                  name=None, reuse=None, batch_size=None, **kwargs):
#         self.hparams = hparams
#         self.w_dims = w_dims
#         self.b_dims = b_dims
#         self.weight_prior = weight_prior
#         self.trainable = trainable
#         self.units = units
#         self.use_bias = use_bias
#         self._name = name
#         self.reuse = reuse
#         self.built_q_network = False
#
#         self.bayesian_loss = None
#
#
