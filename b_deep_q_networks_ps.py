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

from absl import logging

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
                 learning_rate=0.001,
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
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.optimizer = optimizer
        self.grad_clipping = grad_clipping
        self.gamma = gamma
        self.num_bootstrap_heads = num_bootstrap_heads
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
        self.episode = 0
        self.epsilon_list = []

        # Store parameters shapes
        in_size = self.ipt_shape[-1]
        if self.hparams.num_bootstrap_heads:
            out_size = self.hparams.num_bootstrap_heads
        else:
            out_size = 1

        dims = [in_size] + self.hparams.dense_layers + [out_size]
        self.W_dims = []
        self.b_dims = []

        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            self.W_dims.append([dim1, dim2])
            self.b_dims.append(dim2)

        self.built_flag = False
        self.built_number = 0
        self.replay_buffer_filled = False
        self.ep = 0

    def _build_mlp(self, scope=None):
        # Problem with this one is: the first scope, q_fn_vars is perfect, but not the 2nd one
        # Defining the MLP (for when DenseReparam is used)
        # Online
        self.sigma_prior, self.mixture_weights = self.mixture_gen(self.hparams.num_mixtures)
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
                                                                        reuse=True, num_mixtures=None)

            # self.built_flag = True
            self.built_number += 1

    def build(self):
        """Builds the computational graph and training operations."""
        self._build_graph()
        self._build_training_ops()
        self._build_summary_ops()

    def _build_input_placeholder(self):
        """Creates the input placeholders.

    Input placeholders created:
      observations: shape = [batch_size, hparams.fingerprint_length].
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
            # head is the index of the head we want to choose for decison.
            self.head = tf.placeholder(tf.int32, [], name='head')
            # When sample from memory, the batch_size can be fixed, as it is
            # possible to sample any number of samples from memory.
            # state_t is the state at time step t
            self.state_t = tf.placeholder(tf.float32, [None, fingerprint_length], name='state_t')
            # state_tp1 is the state at time step t + 1, tp1 is short for t plus 1.
            self.state_tp1 = [tf.placeholder(tf.float32, [None, fingerprint_length], name='state_tp1_%i' % i) for i in
                              range(batch_size)]
            # done_mask is a {0, 1} tensor indicating whether state_tp1 is the
            # terminal state.
            self.done_mask = tf.placeholder(tf.float32, (batch_size, 1), name='done_mask')
            self.error_weight = tf.placeholder(tf.float32, (batch_size, 1), name='error_weight')
            self.phi_kl_weight = tf.placeholder(tf.float32, (), name='phi_kl_weight')
            self.theta_kl_weight = tf.placeholder(tf.float32, (), name='theta_kl_weight')
            self.thresh = tf.placeholder(tf.int32, (1,), name='thresh')
            self.episode = tf.placeholder(tf.int32, (1,), name='episode')

    def _mlp_online(self, inputs, built_flag, ep, reuse=None):
        outputs = inputs
        op = [0]
        sample_number = np.random.randint(0, self._n_samples)
        eps_out = 0.0
        phi_list = []
        if self._local_reparameterisation:
            if built_flag:
                eps_out = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs, op = self.layers_online['dense_{}'.format(i)](outputs, op, eps_out,
                                                                       built_flag)  # , built_flag, ep)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)
                    op = tf.layers.batch_normalization(op, fused=True, name='bn_1', reuse=reuse)

            outputs, op = self.layers_online['dense_final'](outputs, op, eps_out, built_flag)  # , built_flag, ep)
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, phi_W, phi_b = self.layers_online['dense_{}'.format(i)](outputs, op, None)
                phi_list.append(phi_W)
                phi_list.append(phi_b)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs, op, phi_W, phi_b = self.layers_online['dense_final'](outputs, op, None)
            phi_list.append(phi_W)
            phi_list.append(phi_b)

        return outputs, op, op[sample_number], sample_number, phi_list

    def _mlp_target(self, inputs, built_flag, ep, reuse=None):
        outputs = inputs
        op = [0]
        sample_number = np.random.randint(0, self._n_samples)
        eps_out = 0.0
        if self._local_reparameterisation:
            if built_flag:
                eps_out = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, = self.layers_target['dense_{}'.format(i)](outputs, op, eps_out,
                                                                        built_flag)  # , built_flag, ep)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)
                    op = tf.layers.batch_normalization(op, fused=True, name='bn_1', reuse=reuse)

            outputs, op = self.layers_target['dense_final'](outputs, op, eps_out, built_flag)  # , built_flag, ep)
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs, op, phi_W, phi_b = self.layers_target['dense_{}'.format(i)](outputs, op, None)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                op = getattr(tf.nn, self.hparams.activation)(op)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs, op, phi_W, phi_b = self.layers_target['dense_final'](outputs, op, None)

        return outputs, op, op[sample_number]

    def _mlp_online_theta(self, inputs, post_w, built_flag, ep, reuse=None):
        outputs = inputs
        eps_out = 0.0
        if self._local_reparameterisation:
            if built_flag:
                eps_out = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs = self.layers_online['dense_{}'.format(i)](outputs, eps_out, built_flag)  # , built_flag, ep)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs = self.layers_online['dense_final'](outputs, eps_out, built_flag)  # , built_flag, ep)
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs = self.layers_online['dense_{}'.format(i)].call_theta(outputs, post_w[2 * i], post_w[2 * i + 1],
                                                                              None)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs = self.layers_online['dense_final'].call_theta(outputs, post_w[-2], post_w[-1], None)

        return outputs

    def _mlp_target_theta(self, inputs, post_w, built_flag, ep, reuse=None):
        outputs = inputs
        eps_out = 0.0
        if self._local_reparameterisation:
            if built_flag:
                eps_out = get_random((inputs.shape[0], self.ipt_shape[-1]), avg=0., std=1.)
            for i in range(len(self.hparams.dense_layers)):
                outputs = self.layers_target['dense_{}'.format(i)](outputs, eps_out, built_flag)  # , built_flag, ep)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs = self.layers_target['dense_final'](outputs, eps_out, built_flag)  # , built_flag, ep)
        else:
            for i in range(len(self.hparams.dense_layers)):
                outputs = self.layers_target['dense_{}'.format(i)].call_theta(outputs, post_w[2 * i], post_w[2 * i + 1],
                                                                              None)
                outputs = getattr(tf.nn, self.hparams.activation)(outputs)
                if self.hparams.batch_norm:
                    outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

            outputs = self.layers_target['dense_final'].call_theta(outputs, post_w[-2], post_w[-1], None)

        return outputs

    def eps_out_sample(self, shape):
        eps_out = []
        for s in range(self._n_samples):
            eps_out.append(get_random((shape[0].eval(), self.ipt_shape[-1]), avg=0., std=1.))
        return eps_out

    def bayesian_loss(self, outputs, inputs):
        bayesian_loss = 0.0
        for i in range(len(self.hparams.dense_layers)):
            bayesian_loss += self.layers_online['dense_{}'.format(i)].get_bayesian_loss()
        bayesian_loss += self.layers_online['dense_final'].get_bayesian_loss()
        if all(inputs.shape):
            if isinstance(inputs.shape[0], int):
                bayesian_loss = tf.ones([inputs.shape[0], 1]) * bayesian_loss
        print(bayesian_loss.shape)
        return outputs, bayesian_loss

    def _build_single_q_network(self, observations, head, state_t, state_tp1,
                                done_mask, reward_t, error_weight, phi_kl_weight, theta_kl_weight, episode):
        """Builds the computational graph for a single Q network.

    Briefly, this part is calculating the following two quantities:
    1. q_value = q_fn(observations)
    2. td_error = q_fn(state_t) - reward_t - gamma * q_fn(state_tp1)
    The optimization target is to minimize the td_error.

    This Function uses posterior sharpening to learn a more flexible posterior over weights
    Code inspired by: https://github.com/DeNeutoy/bayesian-rnn/blob/master/bayesian_rnn.py

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
        print('Built status: ' + str(self.built_flag))
        if not self.built_flag:
            print('I am about to build the online mlp: \n')
            self._build_mlp(scope='q_fn')

        # mean, list, sample
        with tf.variable_scope('q_fn'):
            _, _, q_values, _, _ = self._mlp_online(observations, self.built_flag, self.ep)
            q_values = tf.gather(q_values, head, axis=-1)

        # Online Network - calculating q_fn(state_t)
        # The Q network shares parameters with the action graph.
        with tf.variable_scope('q_fn', reuse=True):
            q_t, _, q_t_sample, sample_number, phi_list = self._mlp_online(state_t, self.built_flag, self.ep,
                                                                           reuse=True)

            # Online network parameters
            q_fn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name)
            print('q_fn_vars: ')
            print(q_fn_vars)

            phi_list = []
            for i in range(len(self.hparams.dense_layers)):
                phi_list.extend([q_fn_vars[4 + 6 * i], q_fn_vars[5 + 6 * i]])
            phi_list.extend([q_fn_vars[4 + 6 * 4], q_fn_vars[5 + 6 * 4]])

            phi_parameters_list = []
            for i in range(len(self.hparams.dense_layers)):
                phi_parameters_list.extend(
                    [q_fn_vars[0 + 6 * i], q_fn_vars[1 + 6 * i], q_fn_vars[2 + 6 * i], q_fn_vars[3 + 6 * i]])
            phi_parameters_list.extend(
                [q_fn_vars[0 + 6 * 4], q_fn_vars[1 + 6 * 4], q_fn_vars[2 + 6 * 4], q_fn_vars[3 + 6 * 4]])

        # Means
        phi_means = []
        for i in range(len(self.hparams.dense_layers)):
            phi_means.extend([q_fn_vars[0 + 6 * i], q_fn_vars[2 + 6 * i]])
        phi_means.extend([q_fn_vars[0 + 6 * 4], q_fn_vars[2 + 6 * 4]])

        # Phi_KL Loss
        phi_kl_loss = 0.0
        for i in range(len(self.hparams.dense_layers)):
            phi_kl_loss += self.layers_online['dense_{}'.format(i)].phi_kl_loss(state_t)
        phi_kl_loss += self.layers_online['dense_final'].phi_kl_loss(state_t)
        if all(state_t.shape):
            if isinstance(state_t.shape[0], int):
                phi_kl_loss = tf.ones([state_t.shape[0], 1]) * phi_kl_loss

        if not self.built_flag:
            print('I am about to build the target mlp: \n')
            self._build_mlp(scope='q_tp1')

        # Target Network - calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            if self.hparams.target_type == 'Sample':
                _, _, q_tp1 = zip(
                    *[self._mlp_target(s_tp1, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1])
            elif self.hparams.target_type == 'MAP':
                q_tp1, _, _ = zip(
                    *[self._mlp_target(s_tp1, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1])
            else:
                _, _, q_tp1 = zip(
                    *[self._mlp_target(s_tp1, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1])
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')
        print('Target network variables: ')
        print(q_tp1_vars)
        print()
        phi_tp1_list = []
        for i in range(len(self.hparams.dense_layers)):
            phi_tp1_list.extend([q_tp1_vars[4 + 6 * i], q_tp1_vars[5 + 6 * i]])
        phi_tp1_list.extend([q_tp1_vars[4 + 6 * 4], q_tp1_vars[5 + 6 * 4]])
        print('Target network phi_list: ')
        print(phi_tp1_list)
        print()
        phi_tp1_parameters_list = []
        for i in range(len(self.hparams.dense_layers)):
            phi_tp1_parameters_list.extend(
                [q_tp1_vars[0 + 6 * i], q_tp1_vars[1 + 6 * i], q_tp1_vars[2 + 6 * i], q_tp1_vars[3 + 6 * i]])
        phi_tp1_parameters_list.extend(
            [q_tp1_vars[0 + 6 * 4], q_tp1_vars[1 + 6 * 4], q_tp1_vars[2 + 6 * 4], q_tp1_vars[3 + 6 * 4]])
        if self.double_q:
            with tf.variable_scope('q_fn', reuse=True):
                if self.hparams.doubleq_type == 'Sample':
                    _, _, q_tp1_online, _ = zip(
                        *[self._mlp_online(s_tp1, self.built_flag, self.ep, reuse=True) for s_tp1 in state_tp1])
                elif self.hparams.doubleq_type == 'MAP':
                    q_tp1_online, _, _, _, _ = zip(
                        *[self._mlp_online(s_tp1, self.built_flag, self.ep, reuse=True) for s_tp1 in state_tp1])
            num_heads = 1
            q_tp1_online_idx = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for q in
                                q_tp1_online]
            v_tp1 = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1, q_tp1_online_idx)], axis=0)
        else:
            v_tp1 = tf.stack([tf.reduce_max(q) for q in q_tp1], axis=0)

        q_tp1_masked = tf.multiply((1.0 - done_mask), v_tp1)
        q_t_target = reward_t + self.gamma * q_tp1_masked
        td_target = tf.stop_gradient(q_t_target)
        td_error = td_target - q_t

        # Expected Log-likelihood Huber Loss
        errors = tf.where(tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5, 1.0 * (tf.abs(td_error) - 0.5))
        log_weighted_error = error_weight * errors
        weighted_error = tf.reduce_mean(log_weighted_error)
        logging.info('Finished calculating the negative log likelihood. ')

        # Posterior Sharpening
        with tf.variable_scope('q_fn_theta'):
            posterior_weights, posterior_parameters = self.sharpen_posterior(weighted_error, phi_list)
            q_t_theta = self._mlp_online_theta(state_t, posterior_weights, self.built_flag, self.ep, reuse=True)

        # Summarise the sharpening difference
        normalising_constant = sum([10 * self.W_dims[i][0] * self.W_dims[i][1] for i in range(len(self.W_dims))]) + sum(
            [self.b_dims[i] * 10 for i in range(len(self.b_dims))])
        differences = [tf.cast(phi_means[i], tf.float64) - tf.cast(posterior_parameters[i], tf.float64) for i in
                       range(len(phi_means))]
        total_sharpening_difference = sum([tf.reduce_sum(difference) for difference in differences])
        logging.info('norm const: {}'.format(normalising_constant))
        logging.info('total sd: {}'.format(total_sharpening_difference))
        mean_sharpening_difference = total_sharpening_difference / normalising_constant
        self.sharpening_difference = tf.summary.scalar('sharpening_difference', mean_sharpening_difference)
        q_fn_theta_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn_theta')
        q_fn_vars2 = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn')
        print('q_fn_vars2: ')
        print(q_fn_vars2)

        # Theta Target Network - calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1_theta', reuse=tf.AUTO_REUSE):
            '''if self.hparams.target_type == 'Sample':
                _, _, q_tp1_theta = zip(*[self._mlp_target_theta(s_tp1, posterior_weights, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1])
            elif self.hparams.target_type == 'MAP':
                q_tp1_theta, _, _ = zip(*[self._mlp_target_theta(s_tp1, posterior_weights, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1])
            else:'''
            posterior_weights_tp1, posterior_parameters_tp1 = self.sharpen_posterior(weighted_error, phi_tp1_list)
            q_tp1_theta = [
                self._mlp_target_theta(s_tp1, posterior_weights_tp1, self.built_flag, self.ep, reuse=tf.AUTO_REUSE) for
                s_tp1 in state_tp1]
        q_tp1_theta_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1_theta')
        print('q_tp1_vars_theta: ')
        print(q_tp1_theta_vars)
        print()
        if self.double_q:
            with tf.variable_scope('q_fn_theta', reuse=True):
                '''if self.hparams.doubleq_type == 'Sample':
                    _, _, q_tp1_online_theta, _ = zip(*[self._mlp_online_theta(s_tp1, posterior_weights, self.built_flag, self.ep, reuse=True) for s_tp1 in state_tp1])
                elif self.hparams.doubleq_type == 'MAP':
                    q_tp1_online_theta, _, _, _,  = zip(*[self._mlp_online_theta(s_tp1, posterior_weights, self.built_flag, self.ep, reuse=True) for s_tp1 in state_tp1])
                else:'''
                q_tp1_online_theta = [
                    self._mlp_online_theta(s_tp1, posterior_weights, self.built_flag, self.ep, reuse=True) for s_tp1 in
                    state_tp1]
            num_heads = 1
            q_tp1_online_idx_theta = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for
                                      q in q_tp1_online_theta]
            v_tp1_theta = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1_theta, q_tp1_online_idx_theta)],
                                   axis=0)
        else:
            v_tp1_theta = tf.stack([tf.reduce_max(q) for q in q_tp1_theta], axis=0)

        q_tp1_masked_theta = tf.multiply((1.0 - done_mask), v_tp1_theta)
        q_t_target_theta = reward_t + self.gamma * q_tp1_masked_theta
        td_target_theta = tf.stop_gradient(q_t_target_theta)
        td_error_theta = td_target_theta - q_t_theta

        # Expected Log-likelihood Huber Loss
        errors_theta = tf.where(tf.abs(td_error_theta) < 1.0, tf.square(td_error_theta) * 0.5,
                                1.0 * (tf.abs(td_error_theta) - 0.5))
        log_weighted_error_theta = error_weight * errors_theta
        weighted_error_theta = tf.reduce_mean(log_weighted_error_theta)

        # Bayesian Loss (Expected variational log phi posterior - Expected log phi prior)
        phi_kl_loss = tf.reduce_mean(tf.abs(phi_kl_loss))
        weighted_phi_kl_loss = phi_kl_loss * phi_kl_weight

        # Theta_KL Loss
        theta_kl_loss = 0.0
        for i in range(len(self.hparams.dense_layers)):
            theta_kl_loss += self.layers_online['dense_{}'.format(i)].theta_kl_loss(state_t, posterior_parameters,
                                                                                    phi_means, self.sigma_nought)
        theta_kl_loss += self.layers_online['dense_final'].theta_kl_loss(state_t, posterior_parameters, phi_means,
                                                                         self.sigma_nought)
        if all(state_t.shape):
            if isinstance(state_t.shape[0], int):
                theta_kl_loss = tf.ones([state_t.shape[0], 1]) * theta_kl_loss

        theta_kl_loss = tf.reduce_mean(tf.abs(theta_kl_loss))
        weighted_theta_kl_loss = theta_kl_loss * theta_kl_weight

        # Total Loss
        total_loss = tf.reduce_mean(weighted_phi_kl_loss + weighted_theta_kl_loss + 25 * weighted_error_theta)

        # Perplexities:
        u_plex = tf.reduce_mean(weighted_error)
        s_plex = tf.reduce_mean(weighted_error_theta + weighted_theta_kl_loss)

        # Calculate the mean gradient
        tvars1 = q_fn_vars
        tvars2 = q_fn_vars2
        gradients1 = tf.gradients(td_error, tvars1)
        gradients2 = tf.gradients(td_error, tvars2)
        grads1 = [tf.clip_by_norm(grad, self.hparams.grad_clipping
                                  ) if grad is not None else tf.zeros_like(var, dtype=tf.float32) for grad, var in
                  zip(gradients1, tvars1)]
        grads2 = [tf.clip_by_norm(grad, self.hparams.grad_clipping
                                  ) if grad is not None else tf.zeros_like(var, dtype=tf.float32) for grad, var in
                  zip(gradients2, tvars2)]
        grads_example1 = grads1[0]
        grads_example2 = grads2[0]
        mean_grad1 = tf.reduce_mean(grads_example1)
        mean_grad2 = tf.reduce_mean(grads_example2)
        squared_differences1 = tf.square(grads_example1 - tf.expand_dims(mean_grad1, axis=-1))
        squared_differences2 = tf.square(grads_example2 - tf.expand_dims(mean_grad2, axis=-1))
        grad_var1 = tf.reduce_mean(squared_differences1)
        grad_var2 = tf.reduce_mean(squared_differences2)

        # Log-likelihood variance
        mu_td = tf.reduce_mean(td_error)
        mu_td_theta = tf.reduce_mean(td_error_theta)
        mu_td_centr = tf.square(td_error - tf.expand_dims(mu_td, axis=-1))
        mu_td_centr_theta = tf.square(td_error_theta - tf.expand_dims(mu_td_theta, axis=-1))
        log_like_var = tf.reduce_mean(mu_td_centr)
        log_like_var_theta = tf.reduce_mean(mu_td_centr_theta)

        self.built_flag = True
        print('Built status: ' + str(self.built_flag))

        return (q_values, td_error, td_error_theta, weighted_error, weighted_error_theta, weighted_phi_kl_loss,
                weighted_theta_kl_loss,
                total_loss, u_plex, s_plex, phi_parameters_list, phi_tp1_parameters_list, q_fn_theta_vars,
                q_tp1_theta_vars,
                grad_var1, grad_var2, log_like_var, log_like_var_theta)

    def sharpen_posterior(self, cost, all_weights):
        # inputs are the current batch i.e. state_t with shape [batch_size, hparams.fingerprint_length]
        # phi_weights: (phi_W_mu, phi_W_rho, phi_b_mu, phi_b_rho)
        # Cost is the negative log-likelihood (TD Error): -log(p(y|phi, x), as it's currently calculated.

        logging.info('Calculating gradients: ')
        grads = tf.gradients(cost, all_weights)
        gradients = [grad if grad is not None else tf.zeros_like(weight, dtype=tf.float32) for grad, weight in
                     zip(grads, all_weights)]

        new_weights = []
        new_parameters = []
        # for (weight, grad, scope) in zip(all_weights, gradients, parameter_name_scopes):
        for i in range(int((len(all_weights) - 2) / 2)):
            logging.info('Sharpening for : {}'.format(all_weights[i + i]))
            with tf.variable_scope('dense_{}'.format(i)):
                new_hierarchical_posterior_w, new_posterior_mean_w = self.resample(weight=all_weights[i + i],
                                                                                   grad_ll=gradients[i + i], i=i,
                                                                                   w_type='_W')
                new_weights.append(new_hierarchical_posterior_w)
                new_parameters.append(new_posterior_mean_w)
                new_hierarchical_posterior_b, new_posterior_mean_b = self.resample(weight=all_weights[i + i + 1],
                                                                                   grad_ll=gradients[i + i + 1], i=i,
                                                                                   w_type='_b')
                new_weights.append(new_hierarchical_posterior_b)
                new_parameters.append(new_posterior_mean_b)

        with tf.variable_scope('dense_final'):
            new_hierarchical_posterior_w, new_posterior_mean_w = self.resample(weight=all_weights[-2],
                                                                               grad_ll=gradients[-2], i='final',
                                                                               w_type='_W')
            new_weights.append(new_hierarchical_posterior_w)
            new_parameters.append(new_posterior_mean_w)
            new_hierarchical_posterior_b, new_posterior_mean_b = self.resample(weight=all_weights[-1],
                                                                               grad_ll=gradients[-1], i='final',
                                                                               w_type='_b')
            new_weights.append(new_hierarchical_posterior_b)
            new_parameters.append(new_posterior_mean_b)

        return new_weights, new_parameters

    # @staticmethod
    def resample(self, weight, grad_ll, i, w_type=None):
        # Learning rate per theta; the extra learnable parameter that accounts for conditioning on mini-batch.
        self.sigma_nought = 0.02
        eta = tf.get_variable(name='dense_{}_eta'.format(i) + w_type,
                              shape=weight.get_shape(),
                              initializer=tf.constant_initializer(0.01),
                              trainable=True, dtype=tf.float32)

        new_posterior_mean = tf.stop_gradient(weight) - (eta * grad_ll)
        new_posterior_std = self.sigma_nought * tf.random_normal(weight.get_shape(), mean=0.0, stddev=1.0,
                                                                 dtype=tf.float32)  # Tune the 0.02 value
        new_hierarchical_posterior = new_posterior_mean + new_posterior_std

        return new_hierarchical_posterior, new_posterior_mean

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
            (self.q_values,
             self.td_error,
             self.td_error_theta,
             self.weighted_error,
             self.weighted_error_theta,
             self.weighted_phi_kl_loss,
             self.weighted_theta_kl_loss,
             self.total_loss,
             self.u_plex,
             self.s_plex,
             self.q_fn_vars,
             self.q_tp1_vars,
             self.eta_vars,
             self.eta_tp1_vars,
             self.grad_var1,
             self.grad_var2,
             self.log_like_var,
             self.log_like_var_theta) = self._build_single_q_network(self.observations,
                                                                     self.head,
                                                                     self.state_t,
                                                                     self.state_tp1,
                                                                     self.done_mask,
                                                                     self.reward_t,
                                                                     self.error_weight,
                                                                     self.phi_kl_weight,
                                                                     self.theta_kl_weight,
                                                                     self.episode)
            print('single q network built.' + '\n')
            self.action = self._action_train(self.q_values)

    def _action_train(self, q_vals):  # add a mode here? e.g. IDS during training and greedy during eval
        """Defines the action selection policy during training.

        :param: q_vals:
            The q values of the observations

        """
        return tf.argmax(q_vals)

    def _build_training_ops(self):
        """Creates the training operations.

    Instance attributes created:
      optimization_op: the operation of optimize the loss.
      update_op: the operation to update the q network.
    """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            # Method 1:
            '''tvars = self.q_fn_vars + self.eta_vars 
            grads = [tf.squeeze(tf.clip_by_norm(tf.gradients(self.total_loss, var), self.hparams.grad_clipping), axis=0) for var in tvars]
            learning_rate_decay_fn = functools.partial(tf.train.exponential_decay,
                                                         decay_steps=self.learning_rate_decay_steps,
                                                         decay_rate=self.learning_rate_decay_rate) # change variables here
            self.learning_rate = learning_rate_decay_fn(self.learning_rate, tf.train.get_or_create_global_step())
            #self.lr_update = tf.assign(self.learning_rate, self.new_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimization_op = optimizer.apply_gradients(zip(grads, tvars), 
                                                             global_step=tf.train.get_or_create_global_step(), 
                                                            name='train_step')'''

            # Method 2:
            tvars = self.q_fn_vars + self.eta_vars
            self.optimization_op = contrib_layers.optimize_loss(
                loss=self.total_loss,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                clip_gradients=self.grad_clipping,
                learning_rate_decay_fn=functools.partial(tf.train.exponential_decay,
                                                         decay_steps=self.learning_rate_decay_steps,
                                                         decay_rate=self.learning_rate_decay_rate),
                variables=tvars)

            # Updates the target network with new parameters every 20 episodes (update frequency)
            self.update_op = []
            self.update_op_theta = []
            for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                   sorted(self.q_tp1_vars, key=lambda v: v.name)):
                self.update_op.append(target.assign(var))
            for var, target in zip(sorted(self.eta_vars, key=lambda v: v.name),
                                   sorted(self.eta_tp1_vars, key=lambda v: v.name)):
                self.update_op_theta.append(target.assign(var))
            self.update_op = tf.group(*self.update_op)
            self.update_op_theta = tf.group(*self.update_op_theta)

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
                # self.error_summary = tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(self.td_error)))
                self.weighted_error_summary = tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(self.weighted_error)))
                self.weighted_phi_kl_loss_summary = tf.summary.scalar('phi_kl_loss',
                                                                      tf.reduce_mean(tf.abs(self.weighted_phi_kl_loss)))
                self.weighted_theta_kl_loss_summary = tf.summary.scalar('theta_kl_loss', tf.reduce_mean(
                    tf.abs(self.weighted_theta_kl_loss)))
                self.sharpened_perplexity = tf.summary.scalar('sharpened_perplexity',
                                                              tf.reduce_mean(tf.abs(self.s_plex)))
                self.unsharpened_perplexity = tf.summary.scalar('unsharpened_perplexity',
                                                                tf.reduce_mean(tf.abs(self.u_plex)))
                self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(tf.abs(self.total_loss)))
                self.log_like_var_summary = tf.summary.scalar('log_likelihood_variance',
                                                              tf.reduce_mean(tf.abs(self.log_like_var)))
                self.log_like_var_theta_summary = tf.summary.scalar('log_likelihood_variance',
                                                                    tf.reduce_mean(tf.abs(self.log_like_var_theta)))
                self.grad_var1_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(tf.abs(self.grad_var1)))
                self.grad_var2_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(tf.abs(self.grad_var2)))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                self.episode_summary = tf.summary.merge([smiles_summary, reward_summary])

        self.image_summary = tf.summary.merge_all('IMAGE')

    def get_episode(self, episode):
        ep = np.asscalar(tf.get_default_session().run(self.ep, feed_dict={self.episode: episode}))
        return ep

    def log_result(self, smiles, reward):
        """Summarizes the SMILES string and reward at the end of an episode.

    Args:
      smiles: String. The SMILES string.
      reward: Float. The reward.

    Returns:
      the summary protobuf
    """
        fd = {self.smiles: smiles, self.reward: reward}
        return tf.get_default_session().run(self.episode_summary, feed_dict=fd)

    def _run_action_op(self, observations, head, episode):
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
        self.ep = episode
        # ep = tf.get_default_session().run(self.ep, feed_dict={self.episode: episode})
        action = np.asscalar(
            tf.get_default_session().run(self.action, feed_dict={self.observations: observations, self.head: head}))
        # shape = tf.get_default_session().run(self.out_shape, feed_dict={self.observations: observations, self.head: head})
        return action  # , shape

    def get_action(self,
                   observations,
                   stochastic=True,
                   head=0,
                   episode=None,
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
            action = np.random.randint(0, observations.shape[0])
            return action  # , None
        else:
            return self._run_action_op(observations, head, episode)

    def train(self, states, rewards, next_states, done, weight, phi_kl_weight, theta_kl_weight, ep, summary=True):
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
            ops = [self.td_error, self.td_error_theta, self.weighted_error, self.weighted_error_theta,
                   self.weighted_phi_kl_loss, self.weighted_theta_kl_loss,
                   self.weighted_error_summary, self.weighted_phi_kl_loss_summary, self.weighted_theta_kl_loss_summary,
                   self.total_loss, self.total_loss_summary, self.sharpening_difference,
                   self.unsharpened_perplexity, self.sharpened_perplexity,
                   self.grad_var1_summary, self.log_like_var_summary, self.grad_var2_summary,
                   self.log_like_var_theta_summary,
                   self.optimization_op]
            # ops = [self.td_error, self.error_summary, self.optimization_op]
        else:
            ops = [self.td_error, self, total_loss, self.optimization_op]
        feed_dict = {self.state_t: states,
                     self.reward_t: rewards,
                     self.done_mask: done,
                     self.error_weight: weight,
                     self.phi_kl_weight: phi_kl_weight,
                     self.theta_kl_weight: theta_kl_weight,
                     self.episode: ep}
        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state
        return tf.get_default_session().run(ops, feed_dict=feed_dict)


class MultiObjectiveDeepBBQNetwork(DeepBBQNetwork):
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

    def __init__(self, objective_weight, **kwargs):
        """Creates the model function.

    Args:
      objective_weight: np.array with shape [num_objectives, 1]. The weight
        vector for the objectives.
      **kwargs: arguments for the DeepQNetworks class.

    """
        # Normalize the sum to 1.
        self.objective_weight = objective_weight / np.sum(objective_weight)
        self.num_objectives = objective_weight.shape[0]
        super(MultiObjectiveDeepQNetwork, self).__init__(**kwargs)

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
        batch_size, _ = self.input_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            self.reward_t = tf.placeholder(tf.float32, (batch_size, self.num_objectives), name='reward_t')
            # objective_weight is the weight to scalarize the objective vector:
            # reward = sum (objective_weight_i * objective_i)
            self.objective_weight_input = tf.placeholder(tf.float32, [self.num_objectives, 1], name='objective_weight')

            # split reward for each q network
            rewards_list = tf.split(self.reward_t, self.num_objectives, axis=1)
            q_values_list = []
            self.td_error = []
            self.weighted_error = 0
            self.q_fn_vars = []
            self.q_tp1_vars = []

            # build a Q network for each objective
            for obj_idx in range(self.num_objectives):
                with tf.variable_scope('objective_%i' % obj_idx):
                    (q_values, td_error, weighted_error, q_fn_vars, q_tp1_vars) = self._build_single_q_network(
                        self.observations,
                        self.head, self.state_t,
                        self.state_tp1,
                        self.done_mask,
                        rewards_list[obj_idx],
                        self.error_weight)
                    q_values_list.append(tf.expand_dims(q_values, 1))
                    # td error is for summary only.
                    # weighted error is the optimization goal.
                    self.td_error.append(td_error)
                    self.weighted_error += weighted_error / self.num_objectives
                    self.q_fn_vars += q_fn_vars
                    self.q_tp1_vars += q_tp1_vars
            q_values = tf.concat(q_values_list, axis=1)
            # action is the one that leads to the maximum weighted reward.
            self.action = tf.argmax(
                tf.matmul(q_values, self.objective_weight_input), axis=0)

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
                error_summaries = [tf.summary.scalar('td_error_%i' % i, tf.reduce_mean(tf.abs(self.td_error[i]))) for i
                                   in range(self.num_objectives)]
                self.error_summary = tf.summary.merge(error_summaries)
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.rewards = [tf.placeholder(tf.float32, [], 'summary_reward_obj_%i' % i) for i in
                                range(self.num_objectives)]
                # Weighted sum of the rewards.
                self.weighted_reward = tf.placeholder(tf.float32, [], 'summary_reward_sum')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summaries = [tf.summary.scalar('reward_obj_%i' % i, self.rewards[i]) for i in
                                    range(self.num_objectives)]
                reward_summaries.append(tf.summary.scalar('sum_reward', self.rewards[-1]))
                self.episode_summary = tf.summary.merge([smiles_summary] + reward_summaries)

    def log_result(self, smiles, reward):
        """Summarizes the SMILES string and reward at the end of an episode.

    Args:
      smiles: String. The SMILES string.
      reward: List of Float. The rewards for each objective.

    Returns:
      the summary protobuf.
    """
        feed_dict = {self.smiles: smiles, }
        for i, reward_value in enumerate(reward):
            feed_dict[self.rewards[i]] = reward_value
        # calculated the weighted sum of the rewards.
        feed_dict[self.weighted_reward] = np.asscalar(np.array([reward]).dot(self.objective_weight))
        return tf.get_default_session().run(self.episode_summary, feed_dict=feed_dict)

    def _run_action_op(self, observations, head):
        """Function that runs the op calculating an action given the observations.

    Args:
      observations: np.array. shape = [num_actions, fingerprint_length].
        Observations that can be feed into the Q network.
      head: Integer. The output index to use.

    Returns:
      Integer. which action to be performed.
    """
        return np.asscalar(tf.get_default_session().run(self.action, feed_dict={self.observations: observations,
                                                                                self.objective_weight_input: self.objective_weight,
                                                                                self.head: head}))


class DenseReparameterisation(tf.keras.layers.Layer):
    def __init__(self, hparams, w_dims, b_dims, trainable=None, units=None,
                 use_bias=True, local_reparameterisation=False, n_samples=10,
                 sigma_prior=None, mixture_weights=None,
                 name=None, sample_number=0, reuse=None, **kwargs):
        super(DenseReparameterisation, self).__init__()
        self._name = name
        self._n_samples = 10
        self.n_samples = self.add_variable(shape=[1, ], dtype=tf.float32, trainable=False,
                                           initializer=tf.constant_initializer(n_samples))
        self._local_reparameterisation = local_reparameterisation
        self.use_bias = use_bias
        self.w_dims = w_dims
        self.b_dims = b_dims
        self._batch_size = hparams.batch_size
        self.units = units
        self.reuse = reuse
        self.trainable = trainable
        self._sample_number = sample_number
        # self.log_alpha = 0.0
        self.hparams = hparams
        self.built = False

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

        mixture_weights_norm = statistics.mean(mixture_weights)
        mixture_weights = [m_w / mixture_weights_norm for m_w in mixture_weights]

        self._sigma_prior = sigma_prior
        self._mixture_weights = mixture_weights

        p_s_m = [m_w * np.square(s) for m_w, s in zip(self._mixture_weights, self._sigma_prior)]
        p_s_m = np.sqrt(np.sum(p_s_m))

        self.rho_init = tf.random_uniform_initializer(-4.6, -3.9)

        # Variational parameters
        self.phi_W_mu = None
        self.phi_W_rho = None
        if self.use_bias:
            self.phi_b_mu = None
            self.phi_b_rho = None

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
        if self.phi_W_mu is None:
            self.phi_W_mu = self.add_weight(name=self._name + "_phi_W_mu",
                                            shape=self.w_dims,
                                            dtype=tf.float32,
                                            initializer=tf.keras.initializers.glorot_normal(),
                                            regularizer=None,
                                            trainable=self.trainable,
                                            constraint=None,
                                            partitioner=None,
                                            use_resource=None)

            if self._local_reparameterisation:
                self.log_alpha = self.add_weight(name=self._name + "_alpha",
                                                 shape=(1,),  # self.w_dims,
                                                 dtype=tf.float32,
                                                 initializer=tf.constant_initializer(-4.0),
                                                 regularizer=None,
                                                 trainable=self.trainable,
                                                 constraint=None,
                                                 partitioner=None,
                                                 use_resource=None)
            else:
                self.phi_W_rho = self.add_weight(name=self._name + "_phi_W_rho",
                                                 shape=self.w_dims,
                                                 dtype=tf.float32,
                                                 initializer=self.rho_init,
                                                 regularizer=None,
                                                 trainable=self.trainable,
                                                 # trainable=False,
                                                 constraint=None,
                                                 partitioner=None,
                                                 use_resource=None)
            if self.use_bias:
                self.phi_b_mu = self.add_weight(name=self._name + "_phi_b_mu",
                                                shape=[self.b_dims, ],
                                                dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.01),
                                                regularizer=None,
                                                trainable=self.trainable,
                                                constraint=None,
                                                partitioner=None,
                                                use_resource=None)
                self.phi_b_rho = self.add_weight(name=self._name + "_phi_b_rho",
                                                 shape=[self.b_dims, ],
                                                 dtype=tf.float32,
                                                 initializer=self.rho_init,
                                                 regularizer=None,
                                                 trainable=self.trainable,
                                                 # trainable=False,
                                                 constraint=None,
                                                 partitioner=None,
                                                 use_resource=None)

        self.phi_W = tf.Variable(np.full(self.w_dims, 0.01), name=self._name + '_phi_W', shape=self.w_dims,
                                 dtype=tf.float32)
        self.phi_b = tf.Variable(np.full(self.b_dims, 0.01), name=self._name + '_phi_b', shape=self.b_dims,
                                 dtype=tf.float32)

    def call(self, inputs, inputs_list, eps_out, built_flag=None, ep=None, **kwargs):
        # The forward pass through one layer
        h = 0.0
        h_list = []
        if built_flag:
            self.built = True
        phi_W_list = []
        phi_b_list = []
        # Local Reparameterization
        for s in range(self._n_samples):
            if self._local_reparameterisation:
                if isinstance(inputs_list[0], int):
                    out_mean = tf.matmul(inputs, self.W_mu)
                    out_s = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs, 2.0),
                                                     tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))))
                    if not self.built:
                        out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                    else:
                        out_sigma = tf.multiply(out_s, eps_out)
                    h_sample = out_mean + out_sigma
                    h += h_sample
                else:
                    out_mean = tf.matmul(inputs_list[s], self.W_mu)
                    out_s = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs_list[s], 2.0),
                                                     tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))))
                    if not self.built:
                        out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                    else:
                        out_sigma = tf.multiply(out_s, eps_out)
                    h_sample = out_mean + out_sigma
                    h += h_sample
                h_list.append(h_sample)
                if self.use_bias:
                    b_sample = self.b_mu + tf.multiply(softplus(self.b_rho), self._epsilon_b1_list[s])
                    h += b_sample
                    h_list[s] += b_sample
                h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
                h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]
            # Normal paramtereization
            else:
                W = self.phi_W_mu + tf.multiply(softplus(self.phi_W_rho), self._epsilon_w1_list[s])
                phi_W_list.append(W)
                if isinstance(inputs_list[0], int):
                    h_sample = tf.matmul(inputs, W)
                    h += h_sample
                else:
                    h_sample = tf.matmul(inputs_list[s], W)
                    h += tf.matmul(inputs, W)
                h_list.append(h_sample)
                if self.use_bias:
                    b_sample = self.phi_b_mu + tf.multiply(softplus(self.phi_b_rho), self._epsilon_b1_list[s])
                    phi_b_list.append(b_sample)
                    h += b_sample
                    h_list[s] += b_sample
                h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
                h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]

        phi_W_stacked = tf.cast(tf.stack(phi_W_list), tf.float32)
        phi_b_stacked = tf.cast(tf.stack(phi_b_list), tf.float32)
        # with tf.variable_scope(self._name):
        if built_flag:
            tf.assign(self.phi_W, tf.reduce_sum(phi_W_stacked, axis=0) / self.n_samples)
            tf.assign(self.phi_b, tf.reduce_sum(phi_b_stacked, axis=0) / self.n_samples)
        h_mean = h / self._n_samples
        if not isinstance(eps_out, float):
            return h_mean, h_list, self.phi_W, self.phi_b
        else:
            return h_mean, h_list, self.phi_W, self.phi_b

    def call_theta(self, inputs, theta_W, theta_b, built_flag=None, ep=None, **kwargs):
        # theta_W and theta_b are lists of weight samples from the new hierarchical posterior over theta.
        # The forward pass through one layer
        h = 0.0
        h_list = []
        if built_flag:
            self.built = True
        '''Change the variables here to implement posterior sharpening after normal paramterisation. '''
        # Local Reparameterization
        # for s in range(self._n_samples):
        if self._local_reparameterisation:
            if isinstance(inputs_list[0], int):
                out_mean = tf.matmul(inputs, self.W_mu)
                out_s = tf.sqrt(
                    1e-8 + tf.matmul(tf.pow(inputs, 2.0), tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))))
                if not self.built:
                    out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                else:
                    out_sigma = tf.multiply(out_s, eps_out)
                h_sample = out_mean + out_sigma
                h += h_sample
            else:
                out_mean = tf.matmul(inputs_list[s], self.W_mu)
                out_s = tf.sqrt(1e-8 + tf.matmul(tf.pow(inputs_list[s], 2.0),
                                                 tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))))
                if not self.built:
                    out_sigma = tf.multiply(out_s, self._epsilon_out_list[s])
                else:
                    out_sigma = tf.multiply(out_s, eps_out)
                h_sample = out_mean + out_sigma
                h += h_sample
            h_list.append(h_sample)
            if self.use_bias:
                b_sample = self.b_mu + tf.multiply(softplus(self.b_rho), self._epsilon_b1_list[s])
                h += b_sample
                h_list[s] += b_sample
            h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)
            h_list = [tf.where(tf.is_nan(h), tf.zeros_like(h), h) for h in h_list]
        # Normal paramtereization
        else:
            h = tf.matmul(inputs, theta_W)
            if self.use_bias:
                h += theta_b
            h = tf.where(tf.is_nan(h), tf.zeros_like(h), h)

        h_mean = h / self._n_samples
        return h_mean

    def phi_kl_loss(self, inputs):
        log_prior = 0.0
        log_var_posterior = 0.0
        for s in range(self._n_samples):
            if self._local_reparameterisation:
                W = self.W_mu + tf.multiply(tf.sqrt(tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0))),
                                            self._epsilon_w1_list[s])
            else:
                W = self.phi_W_mu + tf.multiply(softplus(self.phi_W_rho), self._epsilon_w1_list[s])
            log_prior += scale_mixture_prior_generalised(W, self._sigma_prior, self._mixture_weights)
            if self.use_bias:
                b = self.phi_b_mu + tf.multiply(softplus(self.phi_b_rho), self._epsilon_b1_list[s])
                log_prior += scale_mixture_prior_generalised(b, self._sigma_prior, self._mixture_weights)

            if self._local_reparameterisation:
                log_var_posterior += tf.reduce_sum(
                    log_gaussian(W, self.W_mu, tf.sqrt(tf.multiply(tf.exp(self.log_alpha), tf.pow(self.W_mu, 2.0)))))
            else:
                log_var_posterior += tf.reduce_sum(log_gaussian(W, self.phi_W_mu, softplus(self.phi_W_rho)))

            if self.use_bias:
                log_var_posterior += tf.reduce_sum(log_gaussian(b, self.phi_b_mu, softplus(self.phi_b_rho)))

        return (log_var_posterior - log_prior) / self._n_samples

    def theta_kl_loss(self, inputs, theta, phi, sigma_nought):
        log_h_prior = 0.0  # log p(theta | phi)
        log_var_h_post = 0.0  # log q(theta | phi, (x,y))
        if self._local_reparameterisation:
            pass
        else:
            for t in theta:
                log_h_prior += gaussian_prior(t, sigma_nought)
            for (t, phi) in zip(theta, phi):
                log_var_h_post += tf.reduce_sum(log_gaussian(t, phi, sigma_nought))

        return log_var_h_post - log_h_prior

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
        max_steps_per_episode=40,
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=False,
        allowed_ring_sizes=[5, 6],  # [3,4,5,6]
        replay_buffer_size=10000,  # 1000000
        learning_rate=1e-3,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.9,  # 0.8
        num_episodes=5000,
        batch_size=24,  # 64
        learning_frequency=4,
        update_frequency=20,
        grad_clipping=10.0,
        gamma=0.9,
        double_q=True,
        num_bootstrap_heads=0,  # 12
        num_samples=10,
        prioritized=False,
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
        var_range=[-4.6, -3.9],
        target_type='Sample',  # Sample
        doubleq_type='MAP',  # MAP
        local_reparam=False)
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