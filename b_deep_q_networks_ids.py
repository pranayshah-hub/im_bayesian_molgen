# coding=utf-8
# Copyright [2021] [Redacted].

# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python 3

"""DeepQNetwork models for molecule generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import training as contrib_training

from bbb_utils import *


class DeepBBQNetwork(object):
    """Deep Q Network.

  This class implements the network as used in the Nature
  (2015) paper.
  Human-level control through deep reinforcement learning
  https://www.nature.com/articles/nature14236
  https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
  """

    def __init__(self,
                 hparams,
                 ipt_shape,
                 use_bias=True,
                 local_reparameterisation=False,
                 n_samples=10,
                 learning_rate=0.0001,  # 0.001
                 learning_rate_decay_steps=10000,
                 learning_rate_decay_rate=0.8,
                 optimizer='Adam',
                 grad_clipping=None,
                 gamma=1.0,
                 epsilon=0.2,
                 double_q=True,
                 dueling=False,
                 num_bootstrap_heads=0,
                 scope='bdqn',
                 reuse=None,
                 **kwargs):

        super(DeepBBQNetwork, self).__init__()
        """Creates the model function.

    Args:
      input_shape: Tuple. The shape of input.
      q_fn: A function, whose input is the observation features, and the
        output is the Q value of the observation.
      learning_rate: Float. The learning rate of the optimizer.
      learning_rate_decay_steps: Integer. The number of steps between each
        learning rate decay.
      learning_rate_decay_rate: Float. The rate of learning rate decay.
      optimizer: String. Which optimizer to use.
      grad_clipping: Boolean. Whether to clip gradient.
      gamma: Float. Discount factor.
      epsilon: Float. The probability of choosing a random action.
      double_q: Boolean. Whether to use double q learning.
      num_bootstrap_heads: Integer. The number of bootstrap heads to use.
      scope: String or VariableScope. Variable Scope.
      reuse: Boolean or None. Whether or not the variable should be reused.
    """
        self.hparams = hparams
        self.ipt_shape = ipt_shape
        self.learning_rate = self.hparams.learning_rate
        self.learning_rate_decay_steps = self.hparams.learning_rate_decay_steps
        self.learning_rate_decay_rate = self.hparams.learning_rate_decay_rate
        self.optimizer = optimizer
        self.grad_clipping = self.hparams.grad_clipping
        self.gamma = self.hparams.gamma
        self.num_bootstrap_heads = self.hparams.num_bootstrap_heads
        self.double_q = double_q
        self.dueling = dueling
        self.scope = scope
        self.reuse = reuse
        self.epsilon = epsilon

        self.use_bias = use_bias
        self._n_samples = n_samples
        self._local_reparameterisation = local_reparameterisation
        self._batch_size = self.ipt_shape[0]
        self._input_size = hparams.fingerprint_length  # self.ipt_shape[-1]
        self._sample_number = 0
        self.episode = 0.0
        self.num_sigma = 2

        # Store parameters shapes
        in_size = self.ipt_shape[-1]
        if self.hparams.num_bootstrap_heads:
            out_size = self.hparams.num_bootstrap_heads
        else:
            # One for Q-values, one for aleatoric uncertainty
            out_size = 2

        dims = [in_size] + self.hparams.dense_layers + [out_size]
        self.W_dims = []
        self.b_dims = []

        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            self.W_dims.append([dim1, dim2])
            self.b_dims.append(dim2)

        self.built_flag1 = False
        self.built_flag2 = False
        self.built_number = 0
        if self.hparams.multi_obj:
            assert self.built_number <= 4
        else:
            assert self.built_number <= 2

    def _build_mlp(self, scope=None):
        # Online
        if self.hparams.prior_type == 'mixed':
            sp = (float(np.exp(-0.5, dtype=np.float32)),
                  float(np.exp(-2.0, dtype=np.float32)))
        else:
            sp = float(np.exp(-1.373835783, dtype=np.float32))
        if scope == 'q_fn':
            self.layers_online = dict()
            print('I am building the online mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_online['dense_{}'.format(i)] = DenseReparameterisation(hparams=self.hparams,
                                                                                   w_dims=self.W_dims[i],
                                                                                   b_dims=self.b_dims[i],
                                                                                   trainable=True,
                                                                                   local_reparameterisation=self._local_reparameterisation,
                                                                                   sigma_prior=(float(
                                                                                       np.exp(-1.0, dtype=np.float32)),
                                                                                                float(np.exp(-2.0,
                                                                                                             dtype=np.float32))),
                                                                                   mixture_weights=(0.25, 0.75),
                                                                                   name='dense_{}'.format(i),
                                                                                   sample_number=self._sample_number,
                                                                                   batch_size=self._batch_size,
                                                                                   reuse=True, num_mixtures=None)
            self.layers_online['dense_final'] = DenseReparameterisation(hparams=self.hparams,
                                                                        w_dims=self.W_dims[-1],
                                                                        b_dims=self.b_dims[-1],
                                                                        trainable=True,
                                                                        local_reparameterisation=self._local_reparameterisation,
                                                                        sigma_prior=(
                                                                        float(np.exp(-1.0, dtype=np.float32)),
                                                                        float(np.exp(-2.0, dtype=np.float32))),
                                                                        mixture_weights=(0.25, 0.75),
                                                                        name='dense_final',
                                                                        sample_number=self._sample_number,
                                                                        batch_size=self._batch_size,
                                                                        reuse=True, num_mixtures=None)
            self.built_number += 1
        # Target
        elif scope == 'q_tp1':
            self.layers_target = dict()
            print('I am building the target mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_target['dense_{}'.format(i)] = DenseReparameterisation(hparams=self.hparams,
                                                                                   w_dims=self.W_dims[i],
                                                                                   b_dims=self.b_dims[i],
                                                                                   trainable=True,
                                                                                   local_reparameterisation=self._local_reparameterisation,
                                                                                   sigma_prior=(float(
                                                                                       np.exp(-1.0, dtype=np.float32)),
                                                                                                float(np.exp(-2.0,
                                                                                                             dtype=np.float32))),
                                                                                   mixture_weights=(0.25, 0.75),
                                                                                   name='dense_{}'.format(i),
                                                                                   sample_number=self._sample_number,
                                                                                   batch_size=self._batch_size,
                                                                                   reuse=True, num_mixtures=None)
            self.layers_target['dense_final'] = DenseReparameterisation(hparams=self.hparams,
                                                                        w_dims=self.W_dims[-1],
                                                                        b_dims=self.b_dims[-1],
                                                                        trainable=True,
                                                                        local_reparameterisation=self._local_reparameterisation,
                                                                        sigma_prior=(
                                                                        float(np.exp(-1.0, dtype=np.float32)),
                                                                        float(np.exp(-2.0, dtype=np.float32))),
                                                                        mixture_weights=(0.25, 0.75),
                                                                        name='dense_final',
                                                                        sample_number=self._sample_number,
                                                                        batch_size=self._batch_size,
                                                                        reuse=True, num_mixtures=None)

            self.built_number += 1

    def build(self):
        """Builds the computational graph and training operations."""
        self._build_graph()
        self._build_training_ops()
        self._build_summary_ops()

    def _build_input_placeholder(self):
        """Creates the input placeholders.

    Input placeholders created:
      observations: shape = [num_actions, hparams.fingerprint_length].
        The input of the Q function.
      head: shape = [1].
        The index of the head chosen for decision.
      state_t: shape = [batch_size, hparams.fingerprint_length].
        The state at time step t.
      state_tp1: a list of tensors,
        each has shape = [num_actions, hparams.fingerprint_length].
        Note that the num_actions can be different for each tensor.
        The state at time step t+1.
      done_mask: shape = [batch_size, 1]
        Whether state_tp1 is the terminal state.

      error_weight: shape = [batch_size, 1]
        weight for the loss.
    """
        batch_size, fingerprint_length = self.ipt_shape

        with tf.variable_scope(self.scope, reuse=self.reuse):
            # Build the action graph to choose an action.
            # The observations, which are the inputs of the Q function.
            self.observations = tf.placeholder(tf.float32, [None, fingerprint_length], name='observations')
            # self.obs = tf.placeholder(tf.float32, [None, fingerprint_length], name='obs')
            # head is the index of the head we want to choose for decison.
            # See https://arxiv.org/abs/1703.07608
            self.head = tf.placeholder(tf.int32, [], name='head')
            self.aleatoric_head = tf.placeholder(tf.int32, [], name='aleatoric_head')
            self.sample_number = tf.placeholder(tf.int32, [], name='sample_number')
            # When sample from memory, the batch_size can be fixed, as it is
            # possible to sample any number of samples from memory.
            # state_t is the state at time step t
            self.state_t = tf.placeholder(tf.float32, self.ipt_shape, name='state_t')
            # self.state_t_ids = tf.placeholder(tf.float32, self.ipt_shape, name='state_t_ids')
            # state_tp1 is the state at time step t + 1, tp1 is short for t plus 1.
            self.state_tp1 = [tf.placeholder(tf.float32, [None, fingerprint_length], name='state_tp1_%i' % i) for i in
                              range(batch_size)]
            # done_mask is a {0, 1} tensor indicating whether state_tp1 is the
            # terminal state.
            self.done_mask = tf.placeholder(tf.float32, (batch_size, 1), name='done_mask')
            self.error_weight = tf.placeholder(tf.float32, (batch_size, 1), name='error_weight')
            self.kl_weight = tf.placeholder(tf.float32, (), name='kl_weight')
            self.thresh = tf.placeholder(tf.int32, (1,), name='thresh')
            self.episode = tf.placeholder(tf.int32, (1,), name='episode')

    def _mlp_online(self, inputs, built_flag1, built_flag2, reuse=None):
        outputs = inputs
        op = [0]
        sample_number = np.random.randint(0, self._n_samples)
        wl = []
        bl = []
        w_means = []
        b_means = []
        eps_out1 = 0.0
        eps_out2 = 0.0
        if self.hparams.local_reparam:
            if self.hparams.multi_obj:
                if built_flag1 and built_flag2:
                    eps_out1 = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            else:
                if built_flag1:
                    eps_out1 = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, _, _, _, _, _ = self.layers_online['dense_{}'.format(i)](outputs, op, eps_out1, eps_out2,
                                                                                      built_flag1, built_flag2)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            if self.hparams.uq == 'closed':
                outputs, op, e_closed, _, _, _, _ = self.layers_online['dense_final'](outputs, op, eps_out1, eps_out2,
                                                                                      built_flag1, built_flag2)
                return outputs, op, op[sample_number], None, None, None, e_closed
            else:
                outputs, op, _, _, _, _, _ = self.layers_online['dense_final'](outputs, op, eps_out1, eps_out2,
                                                                               built_flag1, built_flag2)
                return outputs, op, op[sample_number], None, None, None, None
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, W_list, b_list, W, b = self.layers_online['dense_{}'.format(i)](outputs, op)
                wl.append(W_list)
                bl.append(b_list)
                w_means.append(W)
                b_means.append(b)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs, op, W_list, b_list, W, b = self.layers_online['dense_final'](outputs, op)
            wl.append(W_list)
            bl.append(b_list)
            w_means.append(W)
            b_means.append(b)

            return outputs, op, op[sample_number], wl, bl, w_means, b_means

    def _mlp_target(self, inputs, built_flag1, built_flag2, reuse=None):
        outputs = inputs
        op = [0]
        sample_number = np.random.randint(0, self._n_samples)
        eps_out1 = 0.0
        eps_out2 = 0.0
        if self.hparams.local_reparam:
            if self.hparams.multi_obj:
                if built_flag1 and built_flag2:
                    eps_out1 = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            else:
                if built_flag1:
                    eps_out1 = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, _, _, _, _, _ = self.layers_target['dense_{}'.format(i)](outputs, op, eps_out1, eps_out2,
                                                                                      built_flag1, built_flag2)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs, op, _, _, _, _, _ = self.layers_target['dense_final'](outputs, op, eps_out1, eps_out2, built_flag1,
                                                                           built_flag2)
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, _, _, _, _ = self.layers_target['dense_{}'.format(i)](outputs, op)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs, op, _, _, _, _ = self.layers_target['dense_final'](outputs, op)

        if self.hparams.uq != 'closed':
            return outputs, op, op[sample_number]
        else:
            return outputs, op, op[sample_number]


    def _build_single_q_network(self, observations, head, aleatoric_head, sample_number, state_t, state_tp1,
                                done_mask, reward_t, error_weight, kl_weight, objective_weight_re=None, ids_reward=None,
                                obj_idx=None):
        """Builds the computational graph for a single Q network.

    Briefly, this part is calculating the following two quantities:
    1. q_value = q_fn(observations)
    2. td_error = q_fn(state_t) - reward_t - gamma * q_fn(state_tp1)
    The optimization target is to minimize the td_error.

    Args:
      observations: shape = [batch_size, hparams.fingerprint_length].
        The input of the Q function.
      head: shape = [1].
        The index of the head chosen for decision in bootstrap DQN.
      state_t: shape = [batch_size, hparams.fingerprint_length].
        The state at time step t.
      state_tp1: a list of tensors, with total number of batch_size,
        each has shape = [num_actions, hparams.fingerprint_length].
        Note that the num_actions can be different for each tensor.
        The state at time step t+1, tp1 is short for t plus 1.
      done_mask: shape = [batch_size, 1]
        Whether state_tp1 is the terminal state.
      reward_t: shape = [batch_size, 1]
        the reward at time step t.
      error_weight: shape = [batch_size, 1]
        weight for the loss.

    Returns:
      q_values: Tensor of [batch_size, 1]. The q values for the observations.
      td_error: Tensor of [batch_size, 1]. The TD error.
      weighted_error: Tensor of [batch_size, 1]. The TD error weighted by
        error_weight.
      total_loss: Tensor of [batch_size, 1]. The ELBO loss objective -
        (expected log variational posterior - expected log prior - expected log likelihood)
      q_fn_vars: List of tf.Variables. The variables of q_fn when computing
        the q_values of state_t
      q_fn_vars: List of tf.Variables. The variables of q_fn when computing
        the q_values of state_tp1

    """
        print('Built status 1: ' + str(self.built_flag1))
        print('Built status 2: ' + str(self.built_flag2))
        if not self.built_flag1:
            print('I am about to build the online mlp: \n')
            self._build_mlp(scope='q_fn')

        if self.hparams.multi_obj:
            if self.built_flag1 and not self.built_flag2:
                print('I am about to build the online mlp for the 2nd dqn: \n')
                self._build_mlp(scope='q_fn')

        # mean, list, sample
        # actions * num_samples
        with tf.variable_scope('q_fn'):
            if self.hparams.uq != 'closed':
                q_vals, q_values_samples, q_vals_sample, w_list, b_list, w, b = self._mlp_online(observations,
                                                                                                 self.built_flag1,
                                                                                                 self.built_flag2)
                q_values = tf.gather(q_vals_sample, head, axis=-1)
                q_values_all = [tf.gather(q_values_samples[i], head, axis=-1) for i in range(len(q_values_samples))]
                q_values_all = tf.stack(q_values_all, axis=1)
                log_pred_var_all = [tf.gather(q_values_samples[i], aleatoric_head, axis=-1) for i in
                                    range(len(q_values_samples))]
                # Calculate mean log_pred_var across all samples (between columns). Left with [num_actions, 1]
                log_pred_var = tf.reduce_mean(tf.stack(log_pred_var_all, axis=1), axis=1)
                # Normalised aleatoric uncertainty: (For reward_eng only)
                pred_var = tf.exp(log_pred_var)
                # Reduce sum over axis=0 over num_actions to get the normalising factor.
                sum_rho2 = tf.reduce_sum(pred_var)
                # Normalised predictive epistemic uncertainty: (For reward_eng only)
                mu_e = tf.reduce_mean(q_values_all, axis=1)
                mu_centralised_e = q_values_all - tf.expand_dims(mu_e, axis=-1)  # axis = -1
                # For each sample, add the contributing mean (action) epistemic uncertainty from each sample for normalising factor
                e_var = tf.reduce_mean(tf.square(mu_centralised_e), axis=1,
                                       keepdims=True)  # This is the average e_uncert per sample
                sum_mu_e = tf.reduce_sum(e_var)
                epistemic_entropy = e_var / (1e-8 + (1 / self._n_samples) * sum_mu_e)
                # Entropy terms
                no_actions = pred_var.shape[0].value
                if no_actions is not None:
                    rho2_norm = pred_var / (1e-8 + (1 / no_actions) * sum_rho2)
                    epistemic_entropy = tf.square(mu_centralised_e) / (1e-10 + e_var)
                else:
                    rho2_norm = pred_var
                # Mean aleatoric predtictive uncertainty for each action: [num_actions, 1]
                aleatoric_entropy = rho2_norm
                # Normalised mean predictive epistemic uncertainty for each action: [num_actions, 1]
                epistemic_entropy = tf.reduce_mean(epistemic_entropy, axis=1)
            else:
                q_vals, q_values_samples, q_vals_sample, _, _, _, e_closed_re = self._mlp_online(observations,
                                                                                                 self.built_flag1,
                                                                                                 self.built_flag2)
                q_values = tf.gather(q_vals_sample, head, axis=-1)
                e_closed_re = tf.gather(e_closed_re, head, axis=-1)
                e_closed_re = e_closed_re / tf.reduce_sum(e_closed_re)
                log_pred_var_all = [tf.gather(q_values_samples[i], aleatoric_head, axis=-1) for i in
                                    range(len(q_values_samples))]
                log_pred_var = tf.reduce_mean(tf.stack(log_pred_var_all, axis=1), axis=1)

        # Online Network - calculating q_fn(state_t)
        # The Q network shares parameters with the action graph.
        with tf.variable_scope('q_fn', reuse=True):
            if self.hparams.uq != 'closed':
                qt, qt_all, _, _, _, _, _ = self._mlp_online(state_t, self.built_flag1, self.built_flag2, reuse=True)
                q_t = tf.expand_dims(tf.gather(qt, head, axis=-1), 1)
                lpv_all = [tf.gather(qt_all[i], aleatoric_head, axis=-1) for i in range(len(qt_all))]
                lpv = tf.reduce_mean(tf.stack(lpv_all, axis=1), axis=1)
                pv = tf.exp(lpv)
                pv_sum = tf.reduce_sum(pv)
                actions = pv.shape[0].value
                if actions is not None:
                    pv_norm = pv / (1e-8 + (1 / actions) * pv_sum)
                else:
                    pv_norm = pv
                a_uncert_norm = pv_norm
                qt_all = [tf.gather(qt_all[i], head, axis=-1) for i in range(len(qt_all))]
                qt_all = tf.stack(qt_all, axis=1)
                # Epistemic  calculation (used to be qt_all)
                mu = tf.reduce_mean(qt_all, axis=1)
                mu_centralised = qt_all - tf.expand_dims(mu, axis=-1)  # axis = -1
                # For summary writing only: (No need to normalise)
                e_uncert = tf.reduce_mean(tf.square(mu_centralised), axis=1)
                # Normalised epistemic uncertainty, type 2
                if w_list is not None:
                    squared_differences_w = [[tf.square(w_list[i][j] - w[i]) for j in range(self._n_samples)] for i in
                                             range(len(w_list))]
                    squared_differences_b = [[tf.square(b_list[i][j] - b[i]) for j in range(self._n_samples)] for i in
                                             range(len(b_list))]
                    normalising_constant = sum(
                        [10 * self.W_dims[i][0] * self.W_dims[i][1] for i in range(len(self.W_dims))]) + sum(
                        [self.b_dims[i] * 10 for i in range(len(self.b_dims))])
                    sum_wb = sum([tf.reduce_sum(tf.stack(squared_differences_w[i], axis=-1)) for i in
                                  range(len(squared_differences_w))])
                    sum_wb += sum([tf.reduce_sum(tf.stack(squared_differences_b[i], axis=-1)) for i in
                                   range(len(squared_differences_b))])
                    epistemic_entropy2 = sum_wb / (normalising_constant + 1e-9)
                    e_uncert2 = epistemic_entropy2
                    norm_sd_w = [
                        [squared_differences_w[i][j] / (1e-9 + epistemic_entropy2) for j in range(self._n_samples)] for
                        i in range(len(w_list))]
                    norm_sd_b = [
                        [squared_differences_b[i][j] / (1e-9 + epistemic_entropy2) for j in range(self._n_samples)] for
                        i in range(len(b_list))]
                    norm_sum = sum([tf.reduce_sum(tf.stack(norm_sd_w[i], axis=-1)) for i in range(len(norm_sd_w))])
                    norm_sum += sum([tf.reduce_sum(tf.stack(norm_sd_b[i], axis=-1)) for i in range(len(norm_sd_b))])
                    epistemic_entropy2_norm = norm_sum / (normalising_constant + 1e-9)
                else:
                    epistemic_entropy2 = tf.constant([0.0], dtype=tf.float32)
                    e_uncert2 = tf.constant([0.0], dtype=tf.float32)
                    epistemic_entropy2_norm = tf.constant([0.0], dtype=tf.float32)
            else:
                qt, _, _, _, _, _, e_closed = self._mlp_online(state_t, self.built_flag1, self.built_flag2, reuse=True)
                e_closed = tf.gather(e_closed, head, axis=-1)
                q_t = tf.expand_dims(tf.gather(qt, head, axis=-1), 1)
                lpv = tf.gather(qt, aleatoric_head, axis=-1)
                pv = tf.exp(lpv)
                pv_sum = tf.reduce_sum(pv)
                actions = pv.shape[0].value
                if actions is not None:
                    pv_norm = pv / (1e-8 + (1 / actions) * pv_sum)
                else:
                    pv_norm = pv
                a_uncert_norm = pv_norm

        bayesian_loss = 0.0
        for i in range(len(self.hparams.dense_layers)):
            bayesian_loss += self.layers_online['dense_{}'.format(i)].get_bayesian_loss(state_t)
        bayesian_loss += self.layers_online['dense_final'].get_bayesian_loss(state_t)
        if all(state_t.shape):
            if isinstance(state_t.shape[0], int):
                bayesian_loss = tf.ones([state_t.shape[0], 1]) * bayesian_loss
                print(bayesian_loss.shape)

        # Online network parameters
        q_fn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn')
        print(q_fn_vars)
        print()
        if not self.built_flag1:
            print('I am about to build the target mlp: \n')
            self._build_mlp(scope='q_tp1')

        if self.hparams.multi_obj:
            if self.built_flag1 and not self.built_flag2:
                print('I am about to build the target mlp for the 2nd dqn: \n')
                self._build_mlp(scope='q_tp1')

        # Target Network - calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            if self.hparams.target_type == 'Sample':
                _, _, q_tp1 = zip(
                    *[self._mlp_target(s_tp1, self.built_flag1, self.built_flag2, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            elif self.hparams.target_type == 'MAP':
                q_tp1, _, _ = zip(
                    *[self._mlp_target(s_tp1, self.built_flag1, self.built_flag2, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            else:
                _, _, q_tp1 = zip(
                    *[self._mlp_target(s_tp1, self.built_flag1, self.built_flag2, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            q_tp1 = [tf.expand_dims(tf.gather(q_tp1_i, head, axis=-1), 1) for q_tp1_i in q_tp1]
        # Target network parameters
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')
        print(q_tp1_vars)

        if self.double_q:
            with tf.variable_scope('q_fn', reuse=True):
                if self.hparams.doubleq_type == 'Sample':
                    _, _, q_tp1_online = zip(
                        *[self._mlp_online(s_tp1, self.built_flag1, self.built_flag2, reuse=True) for s_tp1 in
                          state_tp1])
                elif self.hparams.doubleq_type == 'MAP':
                    q_tp1_online, _, _, _, _, _, _ = zip(
                        *[self._mlp_online(s_tp1, self.built_flag1, self.built_flag2, reuse=True) for s_tp1 in
                          state_tp1])
                q_tp1_online = [tf.expand_dims(tf.gather(q_tp1_online_i, head, axis=-1), 1) for q_tp1_online_i in
                                q_tp1_online]
            num_heads = self.num_bootstrap_heads - 1
            q_tp1_online_idx = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for q in
                                q_tp1_online]
            v_tp1 = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1, q_tp1_online_idx)], axis=0)
        else:
            v_tp1 = tf.stack([tf.reduce_max(q) for q in q_tp1], axis=0)

        q_tp1_masked = (1.0 - done_mask) * v_tp1
        # ids_term = objective_weight_re[obj_idx, 0] * objective_weight_re[obj_idx, -1] * ids_reward # 0.4 * 0.2 * ids_re
        # q_t_target = (reward_t[obj_idx] * objective_weight_re[obj_idx, 0] + ids_term) + self.gamma * q_tp1_masked
        q_t_target = reward_t + self.gamma * q_tp1_masked
        td_target = tf.stop_gradient(q_t_target)
        td_error = td_target - q_t

        # Expected Log-likelihood Huber Loss
        errors = tf.where(tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5, 1.0 * (tf.abs(td_error) - 0.5))
        errors = (errors * tf.exp(-1 * lpv))
        weighted_error = tf.reduce_mean(error_weight * errors + 0.5 * lpv)  # reduce mean converts loss to scalar

        # Bayesian Loss (Expected variational log posterior - Expected log prior)
        bayesian_loss = tf.reduce_mean(tf.abs(bayesian_loss))
        weighted_bayesian_loss = bayesian_loss * kl_weight

        # Total Loss
        total_loss = tf.reduce_mean(weighted_bayesian_loss + weighted_error)  # reduce mean converts loss to scalar
        if self.built_number >= 2:
            self.built_flag1 = True
        if self.built_number == 4:
            self.built_flag2 = True
        print('Built status 1: ' + str(self.built_flag1))
        print('Built status 2: ' + str(self.built_flag2))

        if self.hparams.uq != 'closed' and not self.hparams.multi_obj:
            # Calculate the mean gradient -- finish later
            tvars = q_fn_vars
            grads = [tf.squeeze(tf.clip_by_norm(tf.gradients(td_error, var), self.hparams.grad_clipping), axis=0) for
                     var in
                     tvars]
            grads_example = grads[0]
            mean_grad = tf.reduce_mean(grads_example)
            squared_differences = tf.square(grads_example - tf.expand_dims(mean_grad, axis=-1))
            grad_var = tf.reduce_mean(squared_differences)

            # Log-likelihood variance
            mu_td = tf.reduce_mean(td_error)
            mu_td_centr = tf.square(td_error - tf.expand_dims(mu_td, axis=-1))
            log_like_var = tf.reduce_mean(mu_td_centr)
        elif self.hparams.uq == 'closed' and self.hparams.multi_obj:
            tvars = q_fn_vars
            grads = [tf.squeeze(tf.clip_by_norm(tf.gradients(td_error, var), self.hparams.grad_clipping), axis=0) for
                     var in tvars]
            grads_example = grads[0]
            mean_grad = tf.reduce_mean(grads_example)
            squared_differences = tf.square(grads_example - tf.expand_dims(mean_grad, axis=-1))
            grad_var = tf.reduce_mean(squared_differences)
            mu_td = tf.reduce_mean(td_error)
            mu_td_centr = tf.square(td_error - tf.expand_dims(mu_td, axis=-1))
            log_like_var = tf.reduce_mean(mu_td_centr)

        if self.hparams.uq != 'closed' and not self.hparams.multi_obj:
            return (
            q_values, q_values_all, q_t, qt_all, td_error, w_list, b_list, w, b, weighted_error, weighted_bayesian_loss,
            # q_values_all
            total_loss, q_fn_vars, q_tp1_vars, log_pred_var, aleatoric_entropy, epistemic_entropy,
            epistemic_entropy2, epistemic_entropy2_norm, lpv, a_uncert_norm, e_uncert, e_uncert2, grad_var,
            log_like_var)
        elif self.hparams.uq != 'closed' and self.hparams.multi_obj:
            return (
            q_values, q_values_all, q_t, qt_all, td_error, w_list, b_list, w, b, weighted_error, weighted_bayesian_loss,
            # q_values_all
            total_loss, q_fn_vars, q_tp1_vars, log_pred_var, aleatoric_entropy, epistemic_entropy,
            epistemic_entropy2, epistemic_entropy2_norm, lpv, a_uncert_norm, e_uncert, e_uncert2)
        else:
            return (q_values, q_t, td_error, weighted_error, weighted_bayesian_loss,
                    total_loss, q_fn_vars, q_tp1_vars, log_pred_var, lpv, a_uncert_norm,
                    e_closed_re, e_closed, grad_var, log_like_var)

    def _build_graph(self):
        """Builds the computational graph.

    Input placeholders created:
      reward_t: shape = [batch_size, 1]
        the reward at time step t.

    Instance attributes created:
      q_values: the q values of the observations.
      q_fn_vars: the variables in q function.
      q_tp1_vars: the variables in q_tp1 function.
      td_error: the td_error.
      weighted_error: the weighted td error.
      action: the action to choose next step.
    """
        batch_size, _ = self.ipt_shape
        self.thresh = 50
        with tf.variable_scope(self.scope, reuse=self.reuse):
            # self._build_mlp()
            # self._mlp = MLP(self.hparams) Did not work
            # self._build_prior()
            # print('prior built.' + '\n')
            self._build_input_placeholder()
            print('input placeholder built.' + '\n')
            self.reward_t = tf.placeholder(tf.float32, (batch_size, 1), name='reward_t')
            # The Q network shares parameters with the action graph.
            # tenors start with q or v have shape [batch_size, 1] when not using
            # bootstrap. When using bootstrap, the shapes are [batch_size, num_bootstrap_heads]
            if self.hparams.uq != 'closed':
                (self.q_values,
                 self.q_values_all,
                 self.q_t,
                 self.qt_all,
                 self.td_error,
                 self.w_list,
                 self.b_list,
                 self.w,
                 self.b,
                 self.weighted_error,
                 self.weighted_bayesian_loss,
                 self.total_loss,
                 self.q_fn_vars,
                 self.q_tp1_vars,
                 self.log_pred_var,
                 self.aleatoric_entropy,
                 self.epistemic_entropy,
                 self.epistemic_entropy2,
                 self.epistemic_entropy2_norm,
                 self.aleatoric_uncertainty,
                 self.aleatoric_uncertainty_normalised,
                 self.epistemic_uncertainty,
                 self.epistemic_uncertainty2,
                 self.grad_var,
                 self.log_like_var) = self._build_single_q_network(self.observations,
                                                                   self.head,
                                                                   self.aleatoric_head,
                                                                   self.sample_number,
                                                                   self.state_t,
                                                                   self.state_tp1,
                                                                   self.done_mask,
                                                                   self.reward_t,
                                                                   self.error_weight,
                                                                   self.kl_weight)
            else:
                (self.q_values,
                 self.q_t,
                 self.td_error,
                 self.weighted_error,
                 self.weighted_bayesian_loss,
                 self.total_loss,
                 self.q_fn_vars,
                 self.q_tp1_vars,
                 self.log_pred_var,
                 self.aleatoric_uncertainty,
                 self.aleatoric_uncertainty_normalised,
                 self.e_closed_re,
                 self.e_closed,
                 self.grad_var,
                 self.log_like_var) = self._build_single_q_network(self.observations,
                                                                   self.head,
                                                                   self.aleatoric_head,
                                                                   self.sample_number,
                                                                   self.state_t,
                                                                   self.state_tp1,
                                                                   self.done_mask,
                                                                   self.reward_t,
                                                                   self.error_weight,
                                                                   self.kl_weight)

            print('single q network built.' + '\n')
            self.action_buffer = self._action_train(self.q_values)
            if self.hparams.uq != 'closed':
                self.action = self._action_train_IDS(self.q_values_all,
                                                     self.log_pred_var,
                                                     None,
                                                     self.epistemic_entropy,
                                                     self.epistemic_entropy2,
                                                     self.epistemic_entropy2_norm,
                                                     self.w_list,
                                                     self.b_list,
                                                     self.w,
                                                     self.b)
            else:
                self.action = self._action_train_IDS(self.q_values, self.log_pred_var,
                                                     self.e_closed_re,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None)
            if self.hparams.uq == 'closed':
                (self.ids_value, self.ig, self.e_uncert,
                 _, _, _, self.m_hist,
                 self.s_hist, self.r_hist, self.ig_hist, self.ids_value_hist) = self.ids(self.q_values,
                                                                                         self.log_pred_var,
                                                                                         self.e_closed_re,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None)
                (self.ids_batch, self.ig_batch, self.e_uncert_batch,
                 _, _, _, self.mean_hist,
                 self.std_hist, self.regret_hist, self.inf_gain_hist, self.ids_batch_hist) = self.ids(self.q_t,
                                                                                                      self.aleatoric_uncertainty,
                                                                                                      self.e_closed_re,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None)
            else:
                (self.ids_value, self.ig, self.e_uncert,
                 self.e_uncert_norm, self.e_uncert2, self.e_uncert2_norm, self.m_hist,
                 self.s_hist, self.r_hist, self.ig_hist, self.ids_value_hist) = self.ids(self.q_values_all,
                                                                                         self.log_pred_var,
                                                                                         None,
                                                                                         self.epistemic_entropy,
                                                                                         self.epistemic_entropy2,
                                                                                         self.epistemic_entropy2_norm,
                                                                                         self.w_list,
                                                                                         self.b_list,
                                                                                         self.w,
                                                                                         self.b)
                (self.ids_batch, self.ig_batch, self.e_uncert_batch,
                 self.e_uncert_norm_batch, self.e_uncert2_batch, self.e_uncert2_norm_batch, self.mean_hist,
                 self.std_hist, self.regret_hist, self.inf_gain_hist, self.ids_batch_hist) = self.ids(self.qt_all,
                                                                                                      self.aleatoric_uncertainty,
                                                                                                      None,
                                                                                                      self.epistemic_entropy,
                                                                                                      self.epistemic_entropy2,
                                                                                                      self.epistemic_entropy2_norm,
                                                                                                      self.w_list,
                                                                                                      self.b_list,
                                                                                                      self.w,
                                                                                                      self.b)

    def _action_train(self, q_vals):
        """Defines the action selection policy during training.

        :param: q_vals:
            The q values of the observations

        """
        return tf.argmax(q_vals)

    def _action_train_IDS(self, q_vals, log_pred_var, e_closed, e_uncert_norm, e_uncert2, e_uncert2_norm, w_list=None,
                          b_list=None, w=None, b=None):
        """Defines the action selection during training.
        Based on the IDS Exploration method, code inspired by https://github.com/nikonikolov/rltf/blob/master/rltf/models/dqn_ids.py

        :param: q_vals:
            The q values of the observations

        """
        self.eps = 1e-9
        # Return distribution variance (aleatoric variance)
        self.rho2 = tf.exp(log_pred_var)
        self.sum_rho2 = tf.reduce_sum(self.rho2)
        self.no_actions = self.rho2.shape[0].value

        # Mean (averaging over samples)
        if self.hparams.uq != 'closed':
            # Mean (averaging over samples)
            self.mu = tf.reduce_mean(q_vals, axis=1)

            # Centralise the mean (difference between mean and q-vals)
            self.mu_centralised = q_vals - tf.expand_dims(self.mu, axis=-1)  # axis= -1

            # Variance and Standard Deviation
            self.var = tf.reduce_mean(tf.square(self.mu_centralised), axis=1)
            self.std = tf.sqrt(self.var)

            e_uncert = self.var
            if w_list is not None:
                e_uncert2 = e_uncert2
        else:
            self.var = e_closed
            self.std = tf.sqrt(self.var)

        # Compute the Estimated Regret
        if self.hparams.uq != 'closed':
            self.regret = tf.reduce_max(self.mu + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                    self.mu - self.num_sigma * self.std)
        else:
            self.regret = tf.reduce_max(q_vals + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                    q_vals - self.num_sigma * self.std)

        # Compute the normalised aleatoric variance
        if self.no_actions is not None:
            self.rho2_norm = self.rho2 / (self.eps + (1 / self.no_actions) * self.sum_rho2)
        else:
            self.rho2_norm = self.rho2

        # Compute the Information Gain
        if (self.hparams.reward_eng == 'ids_norm' or
                self.hparams.reward_eng == 'info_gain_norm' or
                self.hparams.reward_eng == 'ids_true_e1_norm' or
                self.hparams.reward_eng == 'info_gain_true_e1_norm'):
            self.inf_gain = tf.log(1 + e_uncert_norm / self.rho2_norm) + self.eps
        elif self.hparams.reward_eng == 'ids_2' or self.hparams.reward_eng == 'info_gain_2':
            self.inf_gain = tf.log(1 + e_uncert2 / self.rho2_norm) + self.eps
        elif self.hparams.reward_eng == 'ids_norm2' or self.hparams.reward_eng == 'info_gain_norm2':
            self.inf_gain = tf.log(1 + e_uncert2_norm / self.rho2_norm) + self.eps
        else:
            # i.e. even if we use e1_norm in reward_eng, we still use e1 in info_gain
            self.inf_gain = tf.log(1 + self.var / self.rho2_norm) + self.eps

        # Compute Regret-Information ratio
        ids_score = tf.square(self.regret) / self.inf_gain
        if self.hparams.reward_eng == 'ids_true_norm' or self.hparams.reward_eng == 'ids_true_e1_norm':
            ids_score = ids_score / (self.eps + tf.reduce_mean(ids_score))
        # Normalise the info gain without using it inside the ids score
        if self.hparams.reward_eng == 'info_gain_true_norm' or 'info_gain_true_e1_norm':
            self.inf_gain = self.inf_gain / (self.eps + tf.reduce_mean(self.inf_gain))

        # Get action that minimises the regret-information score
        # a = tf.argmin(self.ids_score, axis=-1)
        a = tf.argmin(ids_score, axis=0)
        return a

    def get_ids_score(self, state_t, head, aleatoric_head):
        # if state_t.shape[0] != self.hparams.batch_size:
        ids = tf.get_default_session().run(self.ids_value, feed_dict={self.observations: state_t,
                                                                      self.head: head,
                                                                      self.aleatoric_head: aleatoric_head})
        inf_gain = tf.get_default_session().run(self.ig, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head})
        '''if self.hparams.uq != 'closed':                                                       
            m_hist = tf.get_default_session().run(self.m_hist, feed_dict={self.observations: state_t, 
                                                                        self.head: head, 
                                                                        self.aleatoric_head: aleatoric_head})
            s_hist = tf.get_default_session().run(self.s_hist, feed_dict={self.observations: state_t, 
                                                                        self.head: head, 
                                                                        self.aleatoric_head: aleatoric_head})                                                            
            r_hist = tf.get_default_session().run(self.r_hist, feed_dict={self.observations: state_t, 
                                                                        self.head: head, 
                                                                        self.aleatoric_head: aleatoric_head})
            ig_hist = tf.get_default_session().run(self.ig_hist, feed_dict={self.observations: state_t, 
                                                                        self.head: head, 
                                                                        self.aleatoric_head: aleatoric_head})
            ids_value_hist = tf.get_default_session().run(self.ids_value_hist, feed_dict={self.observations: state_t, 
                                                                        self.head: head, 
                                                                        self.aleatoric_head: aleatoric_head})'''
        e1 = tf.get_default_session().run(self.e_uncert, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head})
        if self.hparams.uq != 'closed':
            e1_norm = tf.get_default_session().run(self.e_uncert_norm, feed_dict={self.observations: state_t,
                                                                                  self.head: head,
                                                                                  self.aleatoric_head: aleatoric_head})
            e2 = tf.get_default_session().run(self.e_uncert2, feed_dict={self.observations: state_t,
                                                                         self.head: head,
                                                                         self.aleatoric_head: aleatoric_head})
            e2_norm = tf.get_default_session().run(self.e_uncert2_norm, feed_dict={self.observations: state_t,
                                                                                   self.head: head,
                                                                                   self.aleatoric_head: aleatoric_head})
            return ids, inf_gain, e1, e1_norm, e2, e2_norm, None, None, None, None, None
        else:
            return ids, inf_gain, e1, None, None, None, None, None, None, None, None


    def ids(self, q_vals, log_pred_var, e_closed, e_uncert_norm=None, e_uncert2=None, e_uncert2_norm=None, w_list=None,
            b_list=None, w=None, b=None):
        """Defines the action selection during training.
        Based on the IDS Exploration method,
        code inspired by https://github.com/nikonikolov/rltf/blob/master/rltf/models/dqn_ids.py

        :param: q_vals:
            The q values of the observations
        log_pred_var:
            The log aleatoric uncertainty predicted at the output
        e_closed:
            Closed form epistemic uncertainty
        e_uncert_norm:
            Normalised epistemic uncertainty
        e_uncert2:
            Type 2 epistemic uncertainty
        e_uncert_norm:
            Normalised type 2 epistemic uncertainty

        """
        self.eps = 1e-9
        # Return distribution variance (aleatoric variance)
        self.rho2 = tf.exp(log_pred_var)
        self.sum_rho2 = tf.reduce_sum(self.rho2)
        self.no_actions = self.rho2.shape[0].value

        if self.hparams.uq != 'closed':
            # Mean (averaging over samples)
            self.mu = tf.reduce_mean(q_vals, axis=1)

            # Centralise the mean (difference between mean and q-vals)
            self.mu_centralised = q_vals - tf.expand_dims(self.mu, axis=-1)  # axis= -1

            # Variance and Standard Deviation
            self.var = tf.reduce_mean(tf.square(self.mu_centralised), axis=1)
            self.std = tf.sqrt(self.var)

            e_uncert = self.var
            if w_list is not None:
                e_uncert2 = e_uncert2
        else:
            self.var = e_closed
            self.std = tf.sqrt(self.var)

        # Compute the Estimated Regret
        if self.hparams.uq != 'closed':
            self.regret = tf.reduce_max(self.mu + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                    self.mu - self.num_sigma * self.std)
        else:
            self.regret = tf.reduce_max(q_vals + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                    q_vals - self.num_sigma * self.std)

        # Compute the normalised aleatoric variance
        if self.no_actions is not None:
            self.rho2_norm = self.rho2 / (self.eps + (1 / self.no_actions) * self.sum_rho2)
        else:
            self.rho2_norm = self.rho2

        # Compute the Information Gain
        if (self.hparams.reward_eng == 'ids_norm' or
                self.hparams.reward_eng == 'info_gain_norm' or
                self.hparams.reward_eng == 'ids_true_e1_norm' or
                self.hparams.reward_eng == 'info_gain_true_e1_norm'):
            self.inf_gain = tf.log(1 + e_uncert_norm / self.rho2_norm) + self.eps
        elif self.hparams.reward_eng == 'ids_2' or self.hparams.reward_eng == 'info_gain_2':
            self.inf_gain = tf.log(1 + e_uncert2 / self.rho2_norm) + self.eps
        elif self.hparams.reward_eng == 'ids_norm2' or self.hparams.reward_eng == 'info_gain_norm2':
            self.inf_gain = tf.log(1 + e_uncert2_norm / self.rho2_norm) + self.eps
        else:
            self.inf_gain = tf.log(1 + self.var / self.rho2_norm) + self.eps

        # Compute Regret-Information ratio
        ids_score = tf.square(self.regret) / self.inf_gain
        if self.hparams.reward_eng == 'ids_true_norm' or self.hparams.reward_eng == 'ids_true_e1_norm':
            ids_score = ids_score / (self.eps + tf.reduce_mean(ids_score))
        # Normalise the info gain without using it inside the ids score
        if self.hparams.reward_eng == 'info_gain_true_norm' or 'info_gain_true_e1_norm':
            self.inf_gain = self.inf_gain / (self.eps + tf.reduce_mean(self.inf_gain))

        '''# Add histograms for visualisation
        if self.hparams.uq != 'closed':
            self.mean_hist = tf.summary.histogram("debug/mean", self.mu)
            self.std_hist = tf.summary.histogram("debug/std", self.std)
            self.regret_hist = tf.summary.histogram("debug/regret", self.regret)
            self.inf_gain_hist = tf.summary.histogram("debug/inf_gain", self.inf_gain)
            self.ids_hist = tf.summary.histogram("debug/ids", ids_score)'''

        if self.hparams.uq != 'closed':
            return (ids_score, self.inf_gain, e_uncert, e_uncert_norm, e_uncert2, e_uncert2_norm,
                    None, None, None, None, None)
        else:
            return (ids_score, self.inf_gain, e_closed, None, None, None,
                    None, None, None, None, None)

    def _build_training_ops(self):
        """Creates the training operations.

    Instance attributes created:
      optimization_op: the operation of optimize the loss.
      update_op: the operation to update the q network.
    """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.optimization_op = contrib_layers.optimize_loss(
                loss=self.total_loss,  # has to be a scalar value
                # loss=self.weighted_error,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                clip_gradients=self.grad_clipping,
                learning_rate_decay_fn=functools.partial(tf.train.exponential_decay,
                                                         decay_steps=self.learning_rate_decay_steps,
                                                         decay_rate=self.learning_rate_decay_rate),
                variables=self.q_fn_vars)

            # Updates the target network with new parameters every 20 episodes (update frequency)
            self.update_op = []
            for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                   sorted(self.q_tp1_vars, key=lambda v: v.name)):
                self.update_op.append(target.assign(var))
            self.update_op = tf.group(*self.update_op)

    def _build_summary_ops(self):
        """Creates the summary operations.

    Input placeholders created:
      smiles: the smiles string.
      reward: the reward.

    Instance attributes created:
      error_summary: the operation to log the summary of error.
      episode_summary: the operation to log the smiles string and reward.
    """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                # The td_error here is the difference between q_t and q_t_target.
                # Without abs(), the summary of td_error is actually underestimated.
                # This is the mean td error:
                self.qt_summary = tf.summary.scalar('qt', tf.reduce_mean(self.q_t))
                self.weighted_error_summary = tf.summary.scalar('td_error',
                                                                tf.reduce_mean(tf.abs(self.weighted_error)))
                self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                                                               tf.reduce_mean(tf.abs(self.weighted_bayesian_loss)))
                self.aleatoric_summary = tf.summary.scalar('aleatoric_uncertainty',
                                                           tf.reduce_mean(tf.abs(self.aleatoric_uncertainty)))
                self.aleatoric_normalised_summary = tf.summary.scalar('aleatoric_normalised_uncertainty',
                                                                      tf.reduce_mean(tf.abs(
                                                                          self.aleatoric_uncertainty_normalised)))
                if self.hparams.uq == 'closed':
                    self.epistemic_summary = tf.summary.scalar('epistemic_uncertainty',
                                                               tf.reduce_mean(tf.abs(self.e_closed)))
                else:
                    self.epistemic_summary = tf.summary.scalar('epistemic_uncertainty',
                                                               tf.reduce_mean(tf.abs(self.epistemic_uncertainty)))
                    self.epistemic_summary2 = tf.summary.scalar('epistemic_uncertainty2_norm',
                                                                tf.reduce_mean(tf.abs(self.epistemic_uncertainty2)))
                self.log_like_var_summary = tf.summary.scalar('log_likelihood_variance',
                                                              tf.reduce_mean(tf.abs(self.log_like_var)))
                self.grad_var_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(tf.abs(self.grad_var)))
                self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(tf.abs(self.total_loss)))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                self.reward_adjusted = tf.placeholder(tf.float32, [], 'summary_reward_adjusted')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                reward_summary_adjusted = tf.summary.scalar('reward', self.reward_adjusted)
                self.episode_summary = tf.summary.merge([smiles_summary, reward_summary])
                self.episode_summary_adjusted = tf.summary.merge([smiles_summary, reward_summary_adjusted])

    def get_episode(self, episode):

        return np.asscalar(tf.get_default_session().run(self.episode, feed_dict={self.episode: episode}))

    def log_result(self, smiles, reward, reward_adjusted):
        """Summarizes the SMILES string and reward at the end of an episode.

    Args:
      smiles: String. The SMILES string.
      reward: Float. The reward.

    Returns:
      the summary protobuf
    """
        fd = {self.smiles: smiles, self.reward: reward}
        episode_summary = tf.get_default_session().run(self.episode_summary, feed_dict=fd)
        if reward_adjusted is not None:
            fd_adjusted = {self.smiles: smiles, self.reward_adjusted: reward_adjusted}
            episode_summary_adjusted = tf.get_default_session().run(self.episode_summary_adjusted,
                                                                    feed_dict=fd_adjusted)
            return episode_summary, episode_summary_adjusted
        else:
            return episode_summary

    def get_q_vals(self, observations):

        return tf.get_default_session().run(self.q_values_all, feed_dict={self.observations: observations})

    def _run_action_op(self, observations, head, aleatoric_head, sample_number):
        """Function that runs the op calculating an action given the observations.

    Args:
      observations: np.array. shape = [num_actions, fingerprint_length].
        Observations that can be feed into the Q network.
      head: Integer. The output index to use.

    Returns:
      Integer. which action to be performed.
    """
        # When self.action is run, self.action = tf.argmax(q-values), so the maximising actions is chosen for head:
        # self.head. (Greedy action)
        action = np.asscalar(tf.get_default_session().run(self.action_buffer,
                                                          feed_dict={self.observations: observations, self.head: head,
                                                                     self.sample_number: sample_number}))

        return action

    def get_ids_action(self, obs, head, a_head, update_epsilon):
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        action = np.asscalar(tf.get_default_session().run(self.action, feed_dict={self.observations: obs,
                                                                                  self.head: head,
                                                                                  self.aleatoric_head: a_head}))

        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, obs.shape[0])
        else:
            return action

    def get_action(self,
                   observations,
                   stochastic=False,
                   head=0,
                   aleatoric_head=1,
                   sample_number=None,
                   update_epsilon=None):
        """Function that chooses an action given the observations.

    Args:
      observations: np.array. shape = [num_actions, fingerprint_length].
        Observations that can be feed into the Q network.
      stochastic: Boolean. If set to False all the actions are always
        deterministic (default True).
      head: Integer. The output index to use.
      update_epsilon: Float or None. update epsilon a new value, if None
        no update happens (default: no update).

    Returns:
      Integer. which action to be performed.
    """
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        if stochastic and np.random.uniform() < self.epsilon:
            return np.random.randint(0, observations.shape[0])
        else:
            return self._run_action_op(observations, head, aleatoric_head, sample_number)

    def train(self, states, rewards, next_states, done, weight, kl_weight, ep, head, aleatoric_head, summary=True):
        """Function that takes a transition (s,a,r,s') and optimizes Bellman error.

    Args:
      states: object, a batch of observations.
      rewards: np.array, immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,).
      next_states: object, observations that followed states.
      done: np.array, 1 if obs_t was the last observation in the episode and 0
        otherwise obs_tp1 gets ignored, but must be of the valid shape. dtype
        must be float32 and shape must be (batch_size,).
      weight: np.array, importance sampling weights for every element of the
        batch. dtype must be float32 and shape must be (batch_size,).
      summary: Boolean, whether to get summary.

    Returns:
      td_error: np.array. a list of differences between Q(s,a) and the
        target in Bellman's equation.
        dtype is float32 and shape is (batch_size,).
    """
        if summary:
            if self.hparams.uq == 'closed':
                ops = [self.td_error, self.weighted_error, self.qt_summary,
                       self.weighted_bayesian_loss, self.weighted_error_summary, self.bayesian_loss_summary,
                       self.aleatoric_uncertainty, self.aleatoric_summary, self.aleatoric_normalised_summary,
                       # self.e_closed, self.epistemic_summary,
                       self.epistemic_uncertainty, self.epistemic_summary,
                       self.grad_var_summary, self.log_like_var_summary,
                       self.total_loss, self.total_loss_summary,
                       self.optimization_op]
            else:
                ops = [self.td_error, self.weighted_error, self.qt_summary,
                       self.weighted_bayesian_loss, self.weighted_error_summary, self.bayesian_loss_summary,
                       self.aleatoric_uncertainty, self.aleatoric_summary, self.aleatoric_normalised_summary,
                       self.epistemic_uncertainty, self.epistemic_uncertainty2, self.epistemic_summary,
                       self.epistemic_summary2,
                       self.grad_var_summary, self.log_like_var_summary,
                       self.total_loss, self.total_loss_summary,
                       self.optimization_op]
        else:
            ops = [self.td_error, self, total_loss, self.optimization_op]
        feed_dict = {  # self.obs: states,
            self.observations: states,
            self.state_t: states,
            self.reward_t: rewards,
            self.done_mask: done,
            self.error_weight: weight,
            self.kl_weight: kl_weight,
            self.episode: ep,
            self.head: head,
            self.aleatoric_head: aleatoric_head}
        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state
        return tf.get_default_session().run(ops, feed_dict=feed_dict)


class MultiObjectiveDeepQNetwork(DeepBBQNetwork):
    """Multi Objective Deep Q Network.

    The idea is described in
    Multiobjective Reinforcement Learning: A Comprehensive Overview
    https://ieeexplore.ieee.org/document/6918520/

    Briefly, the difference between this Multi Objective Deep Q Network and
    a naive Deep Q Network is that this one uses one Q network for approximating
    each of the objectives. And a weighted sum of those Q values are used for
    decision making.
    The loss is the summation of the losses of each Q network.
    """

    def __init__(self, hparams, objective_weight, *args, **kwargs):
        """Creates the model function.

        Args:
          objective_weight: np.array with shape [num_objectives, 1]. The weight
            vector for the objectives.
          **kwargs: arguments for the DeepQNetworks class.

        """
        # Normalize the sum to 1.
        self.objective_weight = objective_weight[:2, :] / np.sum(objective_weight[:2, :])
        self.objective_weight_re = self.objective_weight
        if hparams.num_obj == 3:
            self.num_objectives = hparams.num_obj - 1
        else:
            self.num_objectives = objective_weight.shape[0]
        super(MultiObjectiveDeepQNetwork, self).__init__(hparams, *args, **kwargs)

    def _build_mlp(self, scope=None):
        # Problem with this one is: the first scope, q_fn_vars is perfect, but not the 2nd one
        # Defining the MLP (for when DenseReparam is used)
        # Online
        if self.hparams.prior_type == 'mixed':
            sp = (float(np.exp(-1.0, dtype=np.float32)),
                  float(np.exp(-2.0, dtype=np.float32)))
        else:
            sp = float(np.exp(-1.373835783, dtype=np.float32))
        if scope == 'q_fn':
            self.layers_online = dict()
            print('I am building the online mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_online['dense_{}'.format(i)] = DenseReparameterisation(hparams=self.hparams,
                                                                                   w_dims=self.W_dims[i],
                                                                                   b_dims=self.b_dims[i],
                                                                                   trainable=True,
                                                                                   local_reparameterisation=self._local_reparameterisation,
                                                                                   sigma_prior=sp,
                                                                                   mixture_weights=(0.25, 0.75),
                                                                                   name='dense_{}'.format(i),
                                                                                   sample_number=self._sample_number,
                                                                                   batch_size=self._batch_size,
                                                                                   reuse=True, num_mixtures=None)
            self.layers_online['dense_final'] = DenseReparameterisation(hparams=self.hparams,
                                                                        w_dims=self.W_dims[-1],
                                                                        b_dims=self.b_dims[-1],
                                                                        trainable=True,
                                                                        local_reparameterisation=self._local_reparameterisation,
                                                                        sigma_prior=sp,
                                                                        mixture_weights=(0.25, 0.75),
                                                                        name='dense_final',
                                                                        sample_number=self._sample_number,
                                                                        batch_size=self._batch_size,
                                                                        reuse=True, num_mixtures=None)
            self.built_number += 1
        # Target
        elif scope == 'q_tp1':
            self.layers_target = dict()
            print('I am building the target mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_target['dense_{}'.format(i)] = DenseReparameterisation(hparams=self.hparams,
                                                                                   w_dims=self.W_dims[i],
                                                                                   b_dims=self.b_dims[i],
                                                                                   trainable=True,
                                                                                   local_reparameterisation=self._local_reparameterisation,
                                                                                   sigma_prior=sp,
                                                                                   mixture_weights=(0.25, 0.75),
                                                                                   name='dense_{}'.format(i),
                                                                                   sample_number=self._sample_number,
                                                                                   batch_size=self._batch_size,
                                                                                   reuse=True, num_mixtures=None)
            self.layers_target['dense_final'] = DenseReparameterisation(hparams=self.hparams,
                                                                        w_dims=self.W_dims[-1],
                                                                        b_dims=self.b_dims[-1],
                                                                        trainable=True,
                                                                        local_reparameterisation=self._local_reparameterisation,
                                                                        sigma_prior=sp,
                                                                        mixture_weights=(0.25, 0.75),
                                                                        name='dense_final',
                                                                        sample_number=self._sample_number,
                                                                        batch_size=self._batch_size,
                                                                        reuse=True, num_mixtures=None)

            # self.built_flag = True
            self.built_number += 1

    def _build_graph(self):
        """Builds the computational graph.

        Input placeholders created:
          observations: shape = [batch_size, hparams.fingerprint_length].
            The input of the Q function.
          head: shape = [1].
            The index of the head chosen for decision.
          objective_weight: shape = [num_objectives, 1].
            objective_weight is the weight to scalarize the objective vector:
            reward = sum (objective_weight_i * objective_i)
          state_t: shape = [batch_size, hparams.fingerprint_length].
            The state at time step t.
          state_tp1: a list of tensors,
            each has shape = [num_actions, hparams.fingerprint_length].
            Note that the num_actions can be different for each tensor.
            The state at time step t+1.
          done_mask: shape = [batch_size, 1]
            Whether state_tp1 is the terminal state.
          reward_t: shape = [batch_size, num_objectives]
            the reward at time step t.
          error weight: shape = [batch_size, 1]
            weight for the loss.

        Instance attributes created:
          q_values: List of Tensors of [batch_size, 1]. The q values for the
            observations.
          td_error: List of Tensor of [batch_size, 1]. The TD error.
            weighted_error: List of Tensor of [batch_size, 1]. The TD error weighted
            by importance sampling weight.
          q_fn_vars: List of tf.Variables. The variables of q_fn when computing
            the q_values of state_t
          q_fn_vars: List of tf.Variables. The variables of q_fn when computing
            the q_values of state_tp1

        """
        batch_size, _ = self.ipt_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            print('input placeholder built.' + '\n')
            self.reward_t = tf.placeholder(tf.float32, (batch_size, self.num_objectives + 1), name='reward_t')
            # objective_weight is the weight to scalarize the objective vector:
            if self.hparams.learnable_reward:
                self.objective_weight_re = tf.Variable(initial_value=self.hparams.obj_weights, trainable=True,
                                                       shape=[3, 1], name='reward_weights', dtype=tf.float32)
            self.objective_weight_input = tf.placeholder(tf.float32, [self.num_objectives + 1, 1],
                                                         name='objective_weight')

            # split reward for each q network
            # rewards_list = tf.split(self.reward_t[:, :2], self.num_objectives, axis=1)
            rewards_list = tf.split(self.reward_t[:, :2], self.num_objectives, axis=1)
            q_values_list = []
            q_values_all_list = []
            q_t_list = []
            qt_all_list = []
            self.td_error = []
            self.weighted_error = 0
            self.weighted_bayesian_loss = 0
            self.total_loss = 0
            self.q_fn_vars = []
            self.q_tp1_vars = []

            # Uncertainties
            log_pred_var_list = []  # (num_actions, 1)
            aleatoric_entropy_list = []  # (num_actions, 1)
            epistemic_entropy_list = []  # (num_actions, 1)
            epistemic_entropy2_list = []  # ()
            epistemic_entropy2_norm_list = []  # ()
            aleatoric_uncertainty_list = []  # (batch_size, 1)
            aleatoric_uncertainty_normalised_list = []  # (batch_size, 1)
            epistemic_uncertainty_list = []  # (batch_size, 1)
            epistemic_uncertainty2_list = []  # ()

            # build a Q network for each objective
            for obj_idx in range(self.num_objectives):
                with tf.variable_scope('objective_%i' % obj_idx):
                    if self.hparams.uq != 'closed':
                        (q_values,
                         q_values_all,
                         q_t,
                         qt_all,
                         td_error,
                         self.w_list,
                         self.b_list,
                         self.w,
                         self.b,
                         weighted_error,
                         weighted_bayesian_loss,
                         total_loss,
                         q_fn_vars,
                         q_tp1_vars,
                         log_pred_var,
                         aleatoric_entropy,
                         epistemic_entropy,
                         epistemic_entropy2,
                         epistemic_entropy2_norm,
                         aleatoric_uncertainty,
                         aleatoric_uncertainty_normalised,
                         epistemic_uncertainty,
                         epistemic_uncertainty2) = self._build_single_q_network(self.observations,
                                                                                self.head,
                                                                                self.aleatoric_head,
                                                                                self.sample_number,
                                                                                self.state_t,
                                                                                self.state_tp1,
                                                                                self.done_mask,
                                                                                rewards_list,
                                                                                self.error_weight,
                                                                                self.kl_weight)

                    else:
                        (q_values,
                         q_t,
                         td_error,
                         weighted_error,
                         weighted_bayesian_loss,
                         total_loss,
                         q_fn_vars,
                         q_tp1_vars,
                         log_pred_var,
                         aleatoric_uncertainty,
                         aleatoric_uncertainty_normalised,
                         epistemic_entropy,
                         epistemic_uncertainty,
                         grad_var,
                         log_like_var) = self._build_single_q_network(self.observations,
                                                                      self.head,
                                                                      self.aleatoric_head,
                                                                      self.sample_number,
                                                                      self.state_t,
                                                                      self.state_tp1,
                                                                      self.done_mask,
                                                                      rewards_list,
                                                                      self.error_weight,
                                                                      self.kl_weight,
                                                                      self.objective_weight_input)
                    q_values_list.append(tf.expand_dims(q_values, 1))
                    q_t_list.append(tf.expand_dims(q_t, axis=1))
                    if self.hparams.uq != 'closed':
                        q_values_all_list.append(tf.expand_dims(q_values_all, axis=-1))
                        qt_all_list.append(tf.expand_dims(qt_all, axis=-1))
                    self.td_error.append(td_error)
                    self.weighted_error += weighted_error / self.num_objectives
                    self.weighted_bayesian_loss += weighted_bayesian_loss / self.num_objectives
                    self.total_loss += total_loss / self.num_objectives
                    self.q_fn_vars += q_fn_vars
                    self.q_tp1_vars += q_tp1_vars
                    log_pred_var_list.append(tf.expand_dims(log_pred_var, axis=-1))
                    if self.hparams.uq != 'closed':
                        aleatoric_entropy_list.append(tf.expand_dims(aleatoric_entropy, axis=-1))
                        epistemic_entropy2_list.append(
                            tf.expand_dims(tf.expand_dims(epistemic_entropy2, axis=-1), axis=-1))
                        epistemic_entropy2_norm_list.append(
                            tf.expand_dims(tf.expand_dims(epistemic_entropy2_norm, axis=-1), axis=-1))
                    aleatoric_uncertainty_list.append(tf.expand_dims(aleatoric_uncertainty, axis=-1))
                    aleatoric_uncertainty_normalised_list.append(
                        tf.expand_dims(aleatoric_uncertainty_normalised, axis=-1))
                    epistemic_entropy_list.append(tf.expand_dims(epistemic_entropy, axis=-1))
                    epistemic_uncertainty_list.append(tf.expand_dims(epistemic_uncertainty, axis=-1))
                    if self.hparams.uq != 'closed':
                        epistemic_uncertainty2_list.append(
                            tf.expand_dims(tf.expand_dims(epistemic_uncertainty2, axis=-1), axis=-1))

            print('single q networks built.'.format(obj_idx) + '\n')
            self.q_values = tf.concat(q_values_list, axis=1)
            self.q_t = tf.matmul(tf.concat(q_t_list, axis=-1), self.objective_weight_input[:2, :])
            if self.hparams.uq != 'closed':
                self.q_values_all = tf.squeeze(
                    tf.tensordot(tf.concat(q_values_all_list, axis=2), self.objective_weight_input[:2, :], axes=[2, 0]),
                    axis=-1)
                self.qt_all = tf.squeeze(
                    tf.tensordot(tf.concat(qt_all_list, axis=2), self.objective_weight_input[:2, :], axes=[2, 0]),
                    axis=-1)
            # For action-selection/entropies
            self.log_pred_var = tf.squeeze(
                tf.matmul(tf.concat(log_pred_var_list, axis=1), self.objective_weight_input[:2, :]), axis=-1)
            if self.hparams.uq != 'closed':
                self.aleatoric_entropy = tf.squeeze(
                    tf.matmul(tf.concat(aleatoric_entropy_list, axis=1), self.objective_weight_input[:2, :]), axis=-1)
                self.epistemic_entropy2 = tf.squeeze(
                    tf.matmul(tf.concat(epistemic_entropy2_list, axis=1), self.objective_weight_input[:2, :]), -1)
                self.epistemic_entropy2_norm = tf.squeeze(
                    tf.matmul(tf.concat(epistemic_entropy2_norm_list, axis=1), self.objective_weight_input[:2, :]), -1)
            # For the summaries:
            self.aleatoric_uncertainty = tf.squeeze(
                tf.matmul(tf.concat(aleatoric_uncertainty_list, axis=1), self.objective_weight_input[:2, :]), -1)
            self.aleatoric_uncertainty_normalised = tf.squeeze(
                tf.matmul(tf.concat(aleatoric_uncertainty_normalised_list, axis=1), self.objective_weight_input[:2, :]),
                -1)
            self.epistemic_uncertainty = tf.squeeze(
                tf.matmul(tf.concat(epistemic_uncertainty_list, axis=1), self.objective_weight_input[:2, :]), -1)
            self.epistemic_entropy = tf.squeeze(
                tf.matmul(tf.concat(epistemic_entropy_list, axis=1), self.objective_weight_input[:2, :]), -1)
            if self.hparams.uq != 'closed':
                self.epistemic_uncertainty2 = tf.squeeze(
                    tf.matmul(tf.concat(epistemic_uncertainty2_list, axis=1), self.objective_weight_input[:2, :]), -1)
            # action is the one that leads to the maximum weighted reward.
            self.action_buffer = tf.argmax(tf.matmul(self.q_values, self.objective_weight_input[:2, :]), axis=0)

            if self.hparams.uq != 'closed':
                self.action = self._action_train_IDS(self.q_values_all,
                                                     self.log_pred_var,
                                                     None,
                                                     self.epistemic_entropy,
                                                     self.epistemic_entropy2,
                                                     self.epistemic_entropy2_norm,
                                                     self.w_list,
                                                     self.b_list,
                                                     self.w,
                                                     self.b)
            else:
                self.action = self._action_train_IDS(self.q_values, self.log_pred_var,
                                                     self.epistemic_entropy,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None,
                                                     None)
            if self.hparams.uq == 'closed':
                (self.ids_value, self.ig, self.e_uncert,
                 _, _, _, self.m_hist,
                 self.s_hist, self.r_hist, self.ig_hist, self.ids_value_hist) = self.ids(self.q_values,
                                                                                         self.log_pred_var,
                                                                                         self.epistemic_entropy,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None,
                                                                                         None)
                (self.ids_batch, self.ig_batch, self.e_uncert_batch,
                 _, _, _, self.mean_hist,
                 self.std_hist, self.regret_hist, self.inf_gain_hist, self.ids_batch_hist) = self.ids(self.q_t,
                                                                                                      self.aleatoric_uncertainty,
                                                                                                      self.epistemic_entropy,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None,
                                                                                                      None)
            else:
                (self.ids_value, self.ig, self.e_uncert,
                 self.e_uncert_norm, self.e_uncert2, self.e_uncert2_norm, self.m_hist,
                 self.s_hist, self.r_hist, self.ig_hist, self.ids_value_hist) = self.ids(self.q_values_all,
                                                                                         self.log_pred_var,
                                                                                         None,
                                                                                         self.epistemic_entropy,
                                                                                         self.epistemic_entropy2,
                                                                                         self.epistemic_entropy2_norm,
                                                                                         self.w_list,
                                                                                         self.b_list,
                                                                                         self.w,
                                                                                         self.b)
                (self.ids_batch, self.ig_batch, self.e_uncert_batch,
                 self.e_uncert_norm_batch, self.e_uncert2_batch, self.e_uncert2_norm_batch, self.mean_hist,
                 self.std_hist, self.regret_hist, self.inf_gain_hist, self.ids_batch_hist) = self.ids(self.qt_all,
                                                                                                      self.aleatoric_uncertainty,
                                                                                                      None,
                                                                                                      self.epistemic_entropy,
                                                                                                      self.epistemic_entropy2,
                                                                                                      self.epistemic_entropy2_norm,
                                                                                                      self.w_list,
                                                                                                      self.b_list,
                                                                                                      self.w,
                                                                                                      self.b)

    def _build_summary_ops(self):
        """Creates the summary operations.

        Input placeholders created:
          smiles: the smiles string.
          rewards: the rewards.
          weighted_reward: the weighted sum of the rewards.

        Instance attributes created:
          error_summary: the operation to log the summary of error.
          episode_summary: the operation to log the smiles string and reward.
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                # The td_error here is the difference between q_t and q_t_target.
                # Without abs(), the summary of td_error is actually underestimated.
                # Loss Summaries
                error_summaries = [tf.summary.scalar('td_error_%i' % i, tf.reduce_mean(tf.abs(self.td_error[i]))) for i
                                   in range(self.num_objectives)]
                self.qt_summary = tf.summary.scalar('qt', tf.reduce_mean(self.q_t))
                self.weighted_error_summary = tf.summary.merge(error_summaries)
                self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                                                               tf.reduce_mean(tf.abs(self.weighted_bayesian_loss)))
                self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(tf.abs(self.total_loss)))
                self.aleatoric_summary = tf.summary.scalar('aleatoric_uncertainty',
                                                           tf.reduce_mean(tf.abs(self.aleatoric_uncertainty)))
                self.aleatoric_normalised_summary = tf.summary.scalar('aleatoric_normalised_uncertainty',
                                                                      tf.reduce_mean(tf.abs(
                                                                          self.aleatoric_uncertainty_normalised)))
                self.epistemic_summary = tf.summary.scalar('epistemic_uncertainty',
                                                           tf.reduce_mean(tf.abs(self.epistemic_uncertainty)))
                if self.hparams.uq != 'closed':
                    self.epistemic_summary2 = tf.summary.scalar('epistemic_uncertainty2_norm',
                                                                tf.reduce_mean(tf.abs(self.epistemic_uncertainty2)))
                # SMILES and Weighted Reward Summaries
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.rewards = [tf.placeholder(tf.float32, [], 'summary_reward_obj_%i' % i) for i in
                                range(self.num_objectives)]
                self.rewards_adjusted = [tf.placeholder(tf.float32, [], 'summary_reward_adjusted_obj_%i' % i) for i in
                                         range(self.num_objectives)]
                # self.reward_weight_summary = tf.summary.merge([tf.summary.scalar('summary_reward_weight_obj_%i' % i, tf.abs(self.objective_weight_re[i,0])) for i in range(self.num_objectives + 1)])
                # Weighted sum of the rewards.
                self.weighted_reward = tf.placeholder(tf.float32, [], 'summary_reward_sum')
                self.weighted_reward_adjusted = tf.placeholder(tf.float32, [], 'summary_reward_adjusted_sum')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summaries = [tf.summary.scalar('reward_obj_%i' % i, self.rewards[i]) for i in
                                    range(self.num_objectives)]
                reward_adjusted_summaries = [tf.summary.scalar('reward_obj_%i' % i, self.rewards_adjusted[i]) for i in
                                             range(self.num_objectives)]
                # reward_summaries.append(tf.summary.scalar('sum_reward', self.rewards[-1]))
                reward_summaries.append(tf.summary.scalar('sum_reward', self.weighted_reward))
                reward_adjusted_summaries.append(tf.summary.scalar('sum_reward_im', self.weighted_reward_adjusted))
                self.episode_summary = tf.summary.merge([smiles_summary] + reward_summaries)
                self.episode_summary_adjusted = tf.summary.merge([smiles_summary] + reward_adjusted_summaries)

    def get_ids_score(self, state_t, head, aleatoric_head):
        # if state_t.shape[0] != self.hparams.batch_size:
        ids = tf.get_default_session().run(self.ids_value, feed_dict={self.observations: state_t,
                                                                      self.head: head,
                                                                      self.aleatoric_head: aleatoric_head,
                                                                      # self.objective_weight_input: self.objective_weight_re,
                                                                      self.objective_weight_input: self.objective_weight_re})
        inf_gain = tf.get_default_session().run(self.ig, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head,
                                                                    # self.objective_weight_input: self.objective_weight_re,
                                                                    self.objective_weight_input: self.objective_weight_re})
        e1 = tf.get_default_session().run(self.e_uncert, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head,
                                                                    # self.objective_weight_input: self.objective_weight_re,
                                                                    self.objective_weight_input: self.objective_weight_re})
        e1_norm = tf.get_default_session().run(self.e_uncert_norm, feed_dict={self.observations: state_t,
                                                                              self.head: head,
                                                                              self.aleatoric_head: aleatoric_head,
                                                                              # self.objective_weight_input: self.objective_weight_re,
                                                                              self.objective_weight_input: self.objective_weight_re})
        e2 = tf.get_default_session().run(self.e_uncert2, feed_dict={self.observations: state_t,
                                                                     self.head: head,
                                                                     self.aleatoric_head: aleatoric_head,
                                                                     # self.objective_weight_input: self.objective_weight_re,
                                                                     self.objective_weight_input: self.objective_weight_re})
        e2_norm = tf.get_default_session().run(self.e_uncert2_norm, feed_dict={self.observations: state_t,
                                                                               self.head: head,
                                                                               self.aleatoric_head: aleatoric_head,
                                                                               # self.objective_weight_input: self.objective_weight_re,
                                                                               self.objective_weight_input: self.objective_weight_re})
        return ids, inf_gain, e1, e1_norm, e2, e2_norm, None, None, None, None, None


    def log_result(self, smiles, reward, reward_adjusted=None):
        """Summarizes the SMILES string and reward at the end of an episode.

        Args:
          smiles: String. The SMILES string.
          reward: List of Float. The rewards for each objective.

        Returns:
          the summary protobuf.
        """
        feed_dict = {self.smiles: smiles, }
        for i, reward_value in enumerate(reward[:2]):
            feed_dict[self.rewards[i]] = reward_value
        feed_dict[self.weighted_reward] = np.asscalar(
            np.array([reward], dtype=np.float32).dot(self.objective_weight_re))
        # feed_dict[self.weighted_reward] = tf.reduce_sum(tf.tensordot(np.array([reward], dtype=np.float32), self.objective_weight_re, axes=((-1), (0)))).eval()
        episode_summary = tf.get_default_session().run(self.episode_summary, feed_dict=feed_dict)
        if reward_adjusted is not None and self.hparams.multi_obj == 2:
            fd_adjusted = {self.smiles: smiles, }
            for i, reward_value in enumerate(reward_adjusted[:, :2]):
                fd_adjusted[self.rewards_adjusted[i]] = reward_value
            # fd_adjusted[self.weighted_reward_adjusted] = np.asscalar(np.array([reward_adjusted]).dot(self.objective_weight_re))
            fd_adjusted[self.weighted_reward_adjusted] = np.asscalar(
                np.array([reward_adjusted]).dot(self.objective_weight_re))
            episode_summary_adjusted = tf.get_default_session().run(self.episode_summary_adjusted,
                                                                    feed_dict=fd_adjusted)
            return episode_summary, episode_summary_adjusted
        else:
            return episode_summary

    def _run_action_op(self, observations, head, aleatoric_head, sample_number):
        """Function that runs the op calculating an action given the observations.

        Args:
          observations: np.array. shape = [num_actions, fingerprint_length].
            Observations that can be feed into the Q network.
          head: Integer. The output index to use.

        Returns:
          Integer. which action to be performed.
        """
        return np.asscalar(tf.get_default_session().run(self.action_buffer,
                                                        feed_dict={self.observations: observations,
                                                                   # self.objective_weight_input: sess.run(self.objective_weight),
                                                                   self.objective_weight_input: self.objective_weight_re,
                                                                   self.head: head, self.sample_number: sample_number}))

    def get_ids_action(self, obs, head, a_head, update_epsilon):
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        action = np.asscalar(tf.get_default_session().run(self.action, feed_dict={self.observations: obs,
                                                                                  self.head: head,
                                                                                  self.aleatoric_head: a_head}))  # ,
        # self.objective_weight_input: self.objective_weight,
        # self.objective_weight_input: self.objective_weight.eval()}))

        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, obs.shape[0])
        else:
            return action

    def train(self, states, rewards, next_states, done, weight, kl_weight, ep, head, aleatoric_head, summary=True):
        """Function that takes a transition (s,a,r,s') and optimizes Bellman error.

    Args:
      states: object, a batch of observations.
      rewards: np.array, immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,).
      next_states: object, observations that followed states.
      done: np.array, 1 if obs_t was the last observation in the episode and 0
        otherwise obs_tp1 gets ignored, but must be of the valid shape. dtype
        must be float32 and shape must be (batch_size,).
      weight: np.array, importance sampling weights for every element of the
        batch. dtype must be float32 and shape must be (batch_size,).
      summary: Boolean, whether to get summary.

    Returns:
      td_error: np.array. a list of differences between Q(s,a) and the
        target in Bellman's equation.
        dtype is float32 and shape is (batch_size,).
    """
        if summary:
            if self.hparams.uq != 'closed':
                ops = [self.td_error, self.weighted_error, self.qt_summary,
                       self.weighted_bayesian_loss, self.weighted_error_summary, self.bayesian_loss_summary,
                       self.aleatoric_uncertainty, self.aleatoric_summary, self.aleatoric_normalised_summary,
                       self.epistemic_uncertainty, self.epistemic_uncertainty2, self.epistemic_summary,
                       self.epistemic_summary2,
                       # self.objective_weight_re, self.reward_weight_summary,
                       self.total_loss, self.total_loss_summary,
                       self.optimization_op]
            else:
                ops = [self.td_error, self.weighted_error, self.qt_summary,
                       self.weighted_bayesian_loss, self.weighted_error_summary, self.bayesian_loss_summary,
                       self.aleatoric_uncertainty, self.aleatoric_summary, self.aleatoric_normalised_summary,
                       self.epistemic_uncertainty, self.epistemic_summary,
                       self.grad_var_summary, self.log_like_var_summary, self.objective_weight_re,
                       self.reward_weight_summary,
                       self.total_loss, self.total_loss_summary,
                       self.optimization_op]
        else:
            ops = [self.td_error, self, total_loss, self.optimization_op]
        feed_dict = {  # self.obs: states,
            self.observations: states,
            self.state_t: states,
            self.objective_weight_input: self.objective_weight_re,  # .eval() if trainable
            self.reward_t: rewards,  # rewards[:,:2],
            self.done_mask: done,
            self.error_weight: weight,
            self.kl_weight: kl_weight,
            self.episode: ep,
            self.head: head,
            self.aleatoric_head: aleatoric_head}
        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state
        return tf.get_default_session().run(ops, feed_dict=feed_dict)


class DenseReparameterisation(tf.keras.layers.Layer):
    def __init__(self, hparams, w_dims, b_dims, trainable=None, units=None,
                 use_bias=True, local_reparameterisation=False, n_samples=10, sample_number=0,
                 batch_size=None,
                 sigma_prior=None, mixture_weights=None,
                 name=None, reuse=None, **kwargs):
        super(DenseReparameterisation, self).__init__()
        self._name = name
        self._n_samples = n_samples
        self._local_reparameterisation = local_reparameterisation
        self.use_bias = use_bias
        self.w_dims = w_dims
        self.b_dims = b_dims
        self.units = units
        self._batch_size = batch_size
        self.reuse = reuse
        self.trainable = trainable
        self._sample_number = sample_number
        self.hparams = hparams
        self.built = False

        ######################## Initialising prior Gaussian mixture ###########################

        # If there are no mixture weights, initialise at equal mixtures
        if mixture_weights is None:
            mixture_weights = list()
            for i in range(len(sigma_prior)):
                mixture_weights.append(1.0 / len(sigma_prior))

        if self.hparams.prior_type == 'mixed':
            if len(sigma_prior) != len(mixture_weights):
                raise ValueError('Invalid Gaussian Mixture defined. ')

        for i in mixture_weights:
            if i < 0.0:
                raise ValueError('Invalid Mixture Weights. ')

        mixture_weights_norm = statistics.mean(mixture_weights)
        mixture_weights = [m_w / mixture_weights_norm for m_w in mixture_weights]

        self._sigma_prior = sigma_prior
        self._mixture_weights = mixture_weights

        #######################################################################################

        if self.hparams.prior_type == 'mixed':
            p_s_m = [m_w * np.square(s) for m_w, s in zip(self._mixture_weights, self._sigma_prior)]
            p_s_m = np.sqrt(np.sum(p_s_m))

            if hparams.prior_target == 0.218:
                self.rho_max_init = math.log(math.exp(p_s_m / 2) - 1.0)
                self.rho_min_init = math.log(math.exp(p_s_m / 4) - 1.0)
            elif hparams.prior_target == 4.25:
                self.rho_max_init = math.log(math.exp(p_s_m / 212.078) - 1.0)
                self.rho_min_init = math.log(math.exp(p_s_m / 424.93) - 1.0)
            elif hparams.prior_target == 6.555:
                self.rho_max_init = math.log(math.exp(p_s_m / 3265.708106) - 1.0)
                self.rho_min_init = math.log(math.exp(p_s_m / 6507.637711) - 1.0)

        self.rho_init = tf.random_uniform_initializer(-4.6, -3.9)

        # Variational parameters
        self.W_mu = None
        self.W_rho = None
        if self.use_bias:
            self.b_mu = None
            self.b_rho = None

        # Sample epsilon
        self._epsilon_w1_list = []
        self._epsilon_b1_list = []
        self._epsilon_out_list = []

        self.eps_out = tf.placeholder(tf.float32, (None, self.b_dims), name='eps_out')
        self.eps_out_sample = get_random((self._batch_size, self.b_dims), avg=0., std=1.)

        for i in range(self._n_samples):
            if self.use_bias:
                self._epsilon_b1_list.append(get_random((self.b_dims,), avg=0., std=1.))
            if self._local_reparameterisation:
                self._epsilon_out_list.append(get_random((self._batch_size, self.b_dims), avg=0., std=1.))
            self._epsilon_w1_list.append(get_random(tuple(self.w_dims), avg=0., std=1.))

    def build(self, input_shape):
        if self.W_mu is None:
            self.W_mu = self.add_weight(name=self._name + "_W_mu",
                                        # shape=[input_shape[-1], self.units],
                                        shape=self.w_dims,
                                        dtype=tf.float32,
                                        # initializer=tf.truncated_normal_initializer(stddev=1 / np.sqrt(self._input_size)),
                                        initializer=tf.keras.initializers.glorot_normal(),
                                        regularizer=None,
                                        trainable=self.trainable,
                                        constraint=None,
                                        partitioner=None,
                                        use_resource=None)
            # if self._local_reparameterisation:
            if self.hparams.local_reparam == 'layer_wise':
                self.log_alpha = self.add_weight(name=self._name + "_alpha",
                                                 # shape=[input_shape[-1], self.units],
                                                 shape=(1,),  # self.w_dims,
                                                 dtype=tf.float32,
                                                 initializer=tf.constant_initializer(-4.0),
                                                 regularizer=None,
                                                 trainable=self.trainable,
                                                 constraint=None,
                                                 partitioner=None,
                                                 use_resource=None)
            elif self.hparams.local_reparam == 'neuron_wise':
                self.log_alpha = self.add_weight(name=self._name + "_alpha",
                                                 # shape=[input_shape[-1], self.units],
                                                 shape=(self.b_dims,),  # self.w_dims,
                                                 dtype=tf.float32,
                                                 initializer=tf.constant_initializer(-4.0),
                                                 regularizer=None,
                                                 trainable=self.trainable,
                                                 constraint=None,
                                                 partitioner=None,
                                                 use_resource=None)
            else:
                self.W_rho = self.add_weight(name=self._name + "_W_rho",
                                             # shape=[input_shape[-1], self.units],
                                             shape=self.w_dims,
                                             dtype=tf.float32,
                                             # initializer=tf.constant_initializer(0.0),
                                             initializer=self.rho_init,
                                             regularizer=None,
                                             trainable=self.trainable,
                                             # trainable=False,
                                             constraint=None,
                                             partitioner=None,
                                             use_resource=None)
            if self.use_bias:
                self.b_mu = self.add_weight(name=self._name + "_b_mu",
                                            shape=[self.b_dims, ],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.01),
                                            regularizer=None,
                                            trainable=self.trainable,
                                            constraint=None,
                                            partitioner=None,
                                            use_resource=None)
                self.b_rho = self.add_weight(name=self._name + "_b_rho",
                                             shape=[self.b_dims, ],
                                             dtype=tf.float32,
                                             # initializer=tf.constant_initializer(0.0),
                                             initializer=self.rho_init,
                                             regularizer=None,
                                             trainable=self.trainable,
                                             # trainable=False,
                                             constraint=None,
                                             partitioner=None,
                                             use_resource=None)

    def call(self, inputs, inputs_list, eps_out1=None, eps_out2=None, built_flag1=None, built_flag2=None, **kwargs):
        # The forward pass through one layer
        h = 0.0
        out_s = 0.0
        h_list = []
        W_list = []
        b_list = []
        out_s_list = []
        if self.hparams.multi_obj:
            if built_flag1 and built_flag2:
                self.built = True
        else:
            if built_flag1:
                self.built = True
        # Local Reparameterization
        if self.hparams.uq != 'closed':
            if self.hparams.local_reparam:
                for s in range(self._n_samples):
                    if isinstance(inputs_list[0], int):
                        if self.hparams.local_reparam == 'layer_wise':
                            out_mean = tf.matmul(inputs, self.W_mu)
                            out_s = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs, 2.0), tf.multiply(tf.exp(self.log_alpha),
                                                                                              tf.pow(self.W_mu, 2.0))))
                        elif self.hparams.local_reparam == 'neuron_wise':
                            out_mean = tf.matmul(inputs, self.W_mu)
                            out_s = tf.sqrt(1e-8 + tf.multiply(tf.matmul(tf.pow(inputs, 2.0), tf.pow(self.W_mu, 2.0)),
                                                               tf.exp(self.log_alpha)))
                        if not self.built:
                            out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                        else:
                            out_sigma = tf.multiply(out_s, eps_out1)
                        h_sample = out_mean + out_sigma
                        h += h_sample
                    else:
                        if self.hparams.local_reparam == 'layer_wise':
                            out_mean = tf.matmul(inputs_list[s], self.W_mu)
                            out_s = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs_list[s], 2.0),
                                                             tf.multiply(tf.exp(self.log_alpha),
                                                                         tf.pow(self.W_mu, 2.0))))
                        elif self.hparams.local_reparam == 'neuron_wise':
                            out_mean = tf.matmul(inputs_list[s], self.W_mu)
                            out_s = tf.sqrt(
                                1e-8 + tf.multiply(tf.matmul(tf.pow(inputs_list[s], 2.0), tf.pow(self.W_mu, 2.0)),
                                                   tf.exp(self.log_alpha)))
                        if not self.built:
                            out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                        else:
                            out_sigma = tf.multiply(out_s, eps_out1)
                        h_sample = out_mean + out_sigma
                        h += h_sample
                    h_list.append(h_sample)
                    out_s_list.append(out_s)
                    if self.use_bias:
                        b_sigma = softplus(self.b_rho)
                        b_sample = self.b_mu + tf.multiply(b_sigma, self._epsilon_b1_list[s])
                        h += b_sample
                        h_list[s] += b_sample
                        out_s_list[s] += b_sigma
                h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
                h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]
                h_mean = h / self._n_samples
                return h_mean, h_list, None, None, None, None, None
            # Normal parameterization (# shift two tabs right)
            else:
                W = self.W_mu + tf.multiply(softplus(self.W_rho), self._epsilon_w1_list[s])
                W_list.append(W)
                if isinstance(inputs_list[0], int):
                    h_sample = tf.matmul(inputs, W)
                    h += h_sample
                else:
                    h_sample = tf.matmul(inputs_list[s], W)
                    h += tf.matmul(inputs, W)
                h_list.append(h_sample)
                if self.use_bias:
                    b_sample = self.b_mu + tf.multiply(softplus(self.b_rho), self._epsilon_b1_list[s])
                    b_list.append(b_sample)
                    h += b_sample
                    h_list[s] += b_sample
                h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
                h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]
                h_mean = h / self._n_samples
                W_stacked = tf.stack(W_list)
                b_stacked = tf.stack(b_list)
                self.W = tf.reduce_sum(W_stacked, axis=0) / self._n_samples
                self.b = tf.reduce_sum(b_stacked, axis=0) / self._n_samples
                return h_mean, h_list, W_list, b_list, self.W, self.b, None
        else:
            for s in range(self._n_samples):
                if self.hparams.local_reparam:
                    if isinstance(inputs_list[0], int):
                        if self.hparams.local_reparam == 'layer_wise':
                            out_mean = tf.matmul(inputs, self.W_mu)
                            out_s_sample = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs, 2.0),
                                                                    tf.multiply(tf.exp(self.log_alpha),
                                                                                tf.pow(self.W_mu, 2.0))))
                        if not self.built:
                            out_sigma = tf.multiply(out_s_sample, self._epsilon_out_list[s])
                        else:
                            out_sigma = tf.multiply(out_s_sample, eps_out1)
                        h_sample = out_mean + out_sigma
                        h += h_sample
                        out_s += out_s_sample
                    else:
                        if self.hparams.local_reparam == 'layer_wise':
                            out_mean = tf.matmul(inputs_list[s], self.W_mu)
                            out_s_sample = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs_list[s], 2.0),
                                                                    tf.multiply(tf.exp(self.log_alpha),
                                                                                tf.pow(self.W_mu, 2.0))))
                        if not self.built:
                            out_sigma = tf.multiply(out_s_sample, self._epsilon_out_list[s])
                        else:
                            out_sigma = tf.multiply(out_s_sample, eps_out1)
                        h_sample = out_mean + out_sigma
                        h += h_sample
                        out_s += out_s_sample
                    h_list.append(h_sample)
                    out_s_list.append(out_s_sample)
                    if self.use_bias:
                        b_sigma = softplus(self.b_rho)
                        b_sample = self.b_mu + tf.multiply(b_sigma, self._epsilon_b1_list[s])
                        h += b_sample
                        h_list[s] += b_sample
                        out_s += b_sigma
                        out_s_list[s] += b_sigma
                    h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
                    h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]
                    out_s = tf.where(tf.is_nan(out_s), tf.zeros_like(out_s), out_s)
                    out_s_list = [tf.where(tf.is_nan(out_s), tf.zeros_like(out_s), h) for out_s in out_s_list]
                    out_s_list = [tf.square(out_s) for out_s in out_s_list]
                    h_mean = h / self._n_samples
                    out_s_mean = out_s / self._n_samples

            if self.hparams.uq == 'closed' and self._name == 'dense_final':
                # it returns the most recently sampled out_s
                return h, h_list, tf.square(out_s_mean), out_s_list, None, None, None
            else:
                return h, h_list, None, None, None, None, None



    def get_bayesian_loss(self, inputs):
        if self.hparams.bayesian_loss == 'stochastic':
            log_prior = 0.
            log_var_posterior = 0.
            for s in range(self._n_samples):
                if self.hparams.local_reparam == 'layer_wise':
                    W = self.W_mu + tf.multiply(tf.sqrt(tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))),
                                                self._epsilon_w1_list[s])
                elif self.hparams.local_reparam == 'neuron_wise':
                    log_alpha = tf.expand_dims(self.log_alpha, axis=0)
                    log_alpha = tf.tile(log_alpha, [self.W_mu.shape[0], 1])
                    W = self.W_mu + tf.multiply(tf.sqrt(tf.multiply(tf.exp(log_alpha), tf.pow(self.W_mu, 2.0))),
                                                self._epsilon_w1_list[s])
                elif self.hparams.local_reparam == 'zero_mean':
                    W = tf.multiply(softplus(self.W_rho), self._epsilon_w1_list[s])
                else:
                    W = self.W_mu + tf.multiply(softplus(self.W_rho), self._epsilon_w1_list[s])
                if self.hparams.prior_type == 'mixed':
                    log_prior += scale_mixture_prior_generalised(W, self._sigma_prior, self._mixture_weights)
                else:
                    log_prior += tf.reduce_sum(log_gaussian(W, 0.0, self._sigma_prior))

                if self.use_bias:
                    if self.hparams.local_reparam != 'zero_mean':
                        b = self.b_mu + tf.multiply(softplus(self.b_rho), self._epsilon_b1_list[s])
                    else:
                        b = tf.multiply(softplus(self.b_rho), self._epsilon_b1_list[s])
                    if self.hparams.prior_type == 'mixed':
                        log_prior += scale_mixture_prior_generalised(b, self._sigma_prior, self._mixture_weights)
                    else:
                        log_prior += tf.reduce_sum(log_gaussian(b, 0.0, self._sigma_prior))

                if self.hparams.local_reparam == 'layer_wise':
                    log_var_posterior += tf.reduce_sum(log_gaussian(W, self.W_mu, tf.sqrt(
                        tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0)))))
                elif self.hparams.local_reparam == 'neuron_wise':
                    log_alpha = tf.expand_dims(self.log_alpha, axis=0)
                    log_alpha = tf.tile(log_alpha, [self.W_mu.shape[0], 1])
                    log_var_posterior = tf.reduce_sum(
                        log_gaussian(W, self.W_mu, tf.sqrt(tf.multiply(tf.exp(log_alpha), tf.pow(self.W_mu, 2.0)))))
                elif self.hparams.local_reparam == 'zero_mean':
                    log_var_posterior = tf.reduce_sum(log_gaussian(W, 0, softplus(self.W_rho)))
                else:
                    log_var_posterior += tf.reduce_sum(log_gaussian(W, self.W_mu, softplus(self.W_rho)))

                if self.use_bias:
                    if self.hparams.local_reparam != 'zero_mean':
                        log_var_posterior += tf.reduce_sum(log_gaussian(b, self.b_mu, softplus(self.b_rho)))
                    else:
                        log_var_posterior += tf.reduce_sum(log_gaussian(b, 0, softplus(self.b_rho)))

            return (log_var_posterior - log_prior) / self._n_samples

        elif self.hparams.bayesian_loss == 'closed_form':
            if self.hparams.local_reparam == 'layer_wise':
                if self.hparams.prior_type == 'single':
                    c1 = 1.16145124
                    c2 = -1.50204118
                    c3 = 0.58629921
                    sf = self.w_dims[0] * self.w_dims[1] + self.b_dims
                    return -sf * (
                            0.5 * self.log_alpha + c1 * tf.exp(self.log_alpha) + c2 * tf.pow(tf.exp(self.log_alpha),
                                                                                             2) + c3 * tf.pow(
                        tf.exp(self.log_alpha), 3))
                elif self.hparams.prior_type == 'mixed':
                    # loss = None
                    # return loss
                    self.hparams.bayesian_loss = 'stochastic'
                    return self.get_bayesian_loss(inputs)
            # If derivations are successful, put closed-form versions of other parameterizations here.
            elif self.hparams.local_reparam == 'neuron_wise':
                if self.hparams.prior_type == 'single':
                    loss = None
                    return loss
                elif self.hparams.prior_type == 'mixed':
                    # loss = None
                    # return loss
                    self.hparams.bayesian_loss = 'stochastic'
                    return self.get_bayesian_loss(inputs)
            elif self.hparams.log_var_posterior == 'zero_mean':
                if self.hparams.prior_type == 'single':
                    loss = None
                    return loss
                elif self.hparams.prior_type == 'mixed':
                    # loss = None
                    # return loss
                    self.hparams.bayesian_loss = 'stochastic'
                    return self.get_bayesian_loss(inputs)
            else:
                if self.hparams.prior_type == 'single':
                    loss = 0.0  # Fill in later, KL between two gaussians
                    return loss
                elif self.hparams.prior_type == 'mixed':
                    # loss = None
                    # return loss
                    self.hparams.bayesian_loss = 'stochastic'
                    return self.get_bayesian_loss(inputs)

    def select_sample(self, sample_number):
        if sample_number >= self._n_samples:
            raise ValueError("Invalid sample value number.")
        self._sample_number = sample_number



def get_hparams(**kwargs):
    """Get the hyperparameters for the model from a json object.

  Returns:
    A HParams object containing all the hyperparameters.
  """
    hparams = contrib_training.HParams(
        atom_types=['C', 'O', 'N'],
        max_steps_per_episode=40,  # 20 works better with closed form. 20 for moop.
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=False,
        allowed_ring_sizes=[5, 6],  # [3,4,5,6]
        replay_buffer_size=5000,  # 1000000
        learning_rate=1e-4,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.9,  # 0.8
        num_episodes=4000,
        batch_size=12,  # 64
        learning_frequency=4,
        update_frequency=20,
        grad_clipping=10.0,
        gamma=0.9,
        double_q=True,
        num_bootstrap_heads=2,  # 2 heads: one for predictive mean, one for predictive variance
        num_samples=10,
        prioritized=True,
        prioritized_alpha=0.6,
        prioritized_beta=0.4,
        prioritized_epsilon=1e-6,
        fingerprint_radius=3,
        fingerprint_length=2048,
        dense_layers=[1024, 512, 128, 32],
        activation='relu',
        optimizer='Adam',
        batch_norm=False,
        save_frequency=200,  # 1000
        max_num_checkpoints=10,  # 100
        discount_factor=0.7,
        weighting_type=None,
        num_mixtures=2,
        prior_target=4.25,
        prior_type='mixed',  # choose between mixed and single
        exp='thompson',  # thompson or ids
        target_type='Sample',  # Sample
        doubleq_type='MAP',  # MAP
        local_reparam='layer_wise',  # layer_wise, neuron_wise, zero_mean, None
        bayesian_loss='stochastic',
        reward_eng='ids',  # ids norm with e1 norm
        norm=False,
        re=True,
        multi_obj=False,
        rbs=False,
        num_obj=2,
        obj_weights=[[0.2], [0.8]],
        learnable_reward=False,
        uq='stochastic')  # stochastic or closed form
    return hparams.override_from_dict(kwargs)


def get_fingerprint(smiles, hparams):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((hparams.fingerprint_length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((hparams.fingerprint_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, hparams.fingerprint_radius,
                                                        hparams.fingerprint_length)
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def get_fingerprint_with_steps_left(smiles, steps_left, hparams):
    """Get Morgan Fingerprint of a SMILES string with number of steps left.

  If fixing the max num of steps can be taken in a MDP, the MDP is then
  a time-heterogeneous one. Therefore a time dependent policy is needed
  for optimal performance.

  Args:
    smiles: String. The SMILES string of the molecule.
    steps_left: Integer. The number of steps left in the environment.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length + 1]. The fingerprint.
  """
    fingerprint = get_fingerprint(smiles, hparams)
    return np.append(fingerprint, steps_left)