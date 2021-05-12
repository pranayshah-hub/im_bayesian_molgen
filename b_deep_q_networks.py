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

import statistics
import math
import numpy as np
import time
from absl import logging

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import training as contrib_training
from tensorflow.contrib.distributions import Normal

from bbb import DenseReparameterisation, WeightPriorStudent, WeightPriorARD, WeightPriorMOG
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

        self._batch_size = self.ipt_shape[0]
        self._input_size = hparams.fingerprint_length  # self.ipt_shape[-1]
        self._sample_number = 0
        self.episode = 0

        # Store parameters shapes
        in_size = self.ipt_shape[-1]
        if self.hparams.num_bootstrap_heads:
            out_size = self.hparams.num_bootstrap_heads
        else:
            out_size = 1

        dims = [in_size] + self.hparams.dense_layers + [out_size]
        # self.W_dims = []
        # self.b_dims = []
        self.units = []

        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            # self.W_dims.append([dim1, dim2])
            # self.b_dims.append(dim2)
            self.units.append(dim2)

        self.built_flag1 = False
        self.built_flag2 = False
        self.built_number = 0
        self.replay_buffer_filled = False

    def _build_mlp(self, scope=None):
        # Defining the MLP (for when DenseReparam is used)
        # Online
        # self.sigma_prior, self.mixture_weights = self.mixture_gen(self.hparams.num_mixtures)
        if self.hparams.prior_type == 'mixed':
            sp = (float(np.exp(-0.5, dtype=np.float32)),  # -1.0 (original)
                  float(np.exp(-2.0, dtype=np.float32)))
            weight_prior = WeightPriorMOG(sigma_prior=sp,  # Should be list, even if just 1 value.
                                          mixture_weights=(0.25, 0.75),
                                          calculation_type=self.hparams.bayesian_loss)
        elif self.hparams.prior_type == 'single':
            sp = (float(np.exp(-1.373835783, dtype=np.float32)),)
            weight_prior = WeightPriorMOG(sigma_prior=sp,  # Should be list, even if just 1 value.
                                          mixture_weights=(1.0, ),
                                          calculation_type=self.hparams.bayesian_loss)
        elif self.hparams.prior_type == 'ard':
            weight_prior = WeightPriorARD()
        elif self.hparams.prior_type == 'student':
            weight_prior = WeightPriorStudent()
        else:
            raise ValueError("Invalid prior type.")

        if scope == 'q_fn':
            self.layers_online = dict()
            print('I am building the online mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_online['dense_{}'.format(i)] = DenseReparameterisation(units=self.units[i],
                                                                                   weight_prior=weight_prior,
                                                                                   variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                                   use_clt=self.hparams.use_clt,
                                                                                   activation=None,
                                                                                   uncertainty_propagation_type=None,
                                                                                   trainable=True,
                                                                                   use_bias=self.hparams.use_bias,
                                                                                   name=scope + 'dense_{}'.format(i),
                                                                                   reuse=True)
            self.layers_online['dense_final'] = DenseReparameterisation(units=self.units[-1],
                                                                        weight_prior=weight_prior,
                                                                        variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                        use_clt=self.hparams.use_clt,
                                                                        activation=None,
                                                                        uncertainty_propagation_type=None,
                                                                        trainable=True,
                                                                        use_bias=self.hparams.use_bias,
                                                                        name=scope + 'dense_final',
                                                                        reuse=True)
            self.built_number += 1
        # Target
        elif scope == 'q_tp1':
            self.layers_target = dict()
            print('I am building the target mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_target['dense_{}'.format(i)] = DenseReparameterisation(units=self.units[i],
                                                                                   weight_prior=weight_prior,
                                                                                   variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                                   use_clt=self.hparams.use_clt,
                                                                                   activation=None,
                                                                                   uncertainty_propagation_type=None,
                                                                                   trainable=True,
                                                                                   use_bias=self.hparams.use_bias,
                                                                                   name=scope + 'dense_{}'.format(i),
                                                                                   reuse=True)
            self.layers_target['dense_final'] = DenseReparameterisation(units=self.units[-1],
                                                                        weight_prior=weight_prior,
                                                                        variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                        use_clt=self.hparams.use_clt,
                                                                        activation=None,
                                                                        uncertainty_propagation_type=None,
                                                                        trainable=True,
                                                                        use_bias=self.hparams.use_bias,
                                                                        name=scope + 'dense_final',
                                                                        reuse=True)
            self.built_number += 1
        # Thompson
        elif scope == 'q_thomp':
            self.layers_thomp = dict()
            print('I am building the thompson mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_thomp['dense_{}'.format(i)] = DenseReparameterisation(units=self.units[i],
                                                                                  weight_prior=weight_prior,
                                                                                  variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                                  use_clt=self.hparams.use_clt,
                                                                                  activation=None,
                                                                                  uncertainty_propagation_type=None,
                                                                                  trainable=True,
                                                                                  use_bias=self.hparams.use_bias,
                                                                                  name=scope + 'dense_{}'.format(i),
                                                                                  reuse=True)
            self.layers_thomp['dense_final'] = DenseReparameterisation(units=self.units[-1],
                                                                       weight_prior=weight_prior,
                                                                       variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                       use_clt=self.hparams.use_clt,
                                                                       activation=None,
                                                                       uncertainty_propagation_type=None,
                                                                       trainable=True,
                                                                       use_bias=self.hparams.use_bias,
                                                                       name=scope + 'dense_final',
                                                                       reuse=True)
            self.built_number += 1

    def build(self):
        """Builds the computational graph and training operations."""
        # tf.random.set_random_seed(0) # 1234
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
            self.kl_weight = tf.placeholder(tf.float32, (), name='kl_weight')
            self.thresh = tf.placeholder(tf.int32, (1,), name='thresh')
            self.episode = tf.placeholder(tf.int32, (1,), name='episode')
            self.sn = tf.placeholder(tf.float32, (), name='sn')

    def _mlp_online(self, inputs, layers_dict, reuse=None):
        # Do this for the first layer.
        out_mean, out_samples, out_std = layers_dict['dense_{}'.format(0)](inputs=inputs,
                                                                           training=True,
                                                                           n_samples=self.hparams.n_samples,
                                                                           sample_type="Sample")

        out_samples = [getattr(tf.nn, self.hparams.activation)(h) for h in out_samples]
        out_mean = getattr(tf.nn, self.hparams.activation)(out_mean)

        # if self.hparams.batch_norm:
        #     outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

        for i in range(1, len(self.hparams.dense_layers)):
            new_outputs = list()

            for out_sample in out_samples:
                out_mean, out_samples, out_std = layers_dict['dense_{}'.format(i)](inputs=out_sample,
                                                                                   training=True,
                                                                                   n_samples=1,
                                                                                   sample_type="Sample")  # , built_flag, ep)
                out_samples = [getattr(tf.nn, self.hparams.activation)(h) for h in out_samples]
                out_mean = getattr(tf.nn, self.hparams.activation)(out_mean)

                new_outputs.append(out_samples[0])

            out_samples = new_outputs

            # if self.hparams.batch_norm:
            #     outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)

        new_means = list()
        new_stds = list()
        for out_sample in out_samples:
            out_mean, out_samples, out_std = layers_dict['dense_final'](inputs=out_sample,
                                                                        training=True,
                                                                        n_samples=1,
                                                                        sample_type="Sample")  # , built_flag, ep)
            new_means.append(out_mean)
            new_stds.append(out_std)

        mean_sum = new_means[0]
        std_sum = new_stds[0]
        for out_mean, out_std in zip(new_means[1:],
                                     new_stds[1:]):
            mean_sum = mean_sum + out_mean
            std_sum = std_sum + out_std
        out_mean = mean_sum / len(new_means)
        out_std = std_sum / len(new_stds)

        # for i in range(len(self.hparams.dense_layers)):
        #     out_mean, out_samples, out_std = layers_dict['dense_{}'.format(i)](inputs=outputs,
        #                                                                        training=True,
        #                                                                        n_samples=1,
        #                                                                        sample_type="Sample")
        #     outputs = getattr(tf.nn, self.hparams.activation)(out_samples[0])
        #     if self.hparams.batch_norm:
        #         outputs = tf.layers.batch_normalization(outputs, fused=True, name='bn_1', reuse=reuse)
        #
        # out_mean, out_samples, out_std = layers_dict['dense_final'](inputs=outputs,
        #                                                             training=True,
        #                                                             n_samples=1,
        #                                                             sample_type="Sample")

        return out_mean, new_means, out_mean

    def _build_single_q_network(self, observations, head, state_t, state_tp1,
                                done_mask, reward_t, error_weight, kl_weight):  # add sn if changing TS frequency
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
      kl_weight: shape = [1].
        weight for the KL loss.

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
        if not self.hparams.thomp_freq == 20:
            if not self.built_flag1:
                print('I am about to build the online mlp: \n')
                self._build_mlp(scope='q_fn')

            if self.hparams.multi_obj:
                if self.built_flag1 and not self.built_flag2:
                    print('I am about to build the online mlp for the 2nd dqn: \n')
                    self._build_mlp(scope='q_fn')

            # mean, list, sample
            with tf.variable_scope('q_fn'):
                _, _, q_values = self._mlp_online(observations, self.layers_online)
                q_values = tf.gather(q_values, head, axis=-1)

        else:
            if not self.built_flag1:
                print('I am about to build the thompson mlp: \n')
                self._build_mlp(scope='q_thomp')

            with tf.variable_scope('q_thomp'):
                _, _, q_values = self._mlp_online(observations, self.layers_thomp)
                q_values = tf.gather(q_values, head, axis=-1)

            q_thomp_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_thomp')

        # Online Network - calculating q_fn(state_t)
        # The Q network shares parameters with the action graph.
        with tf.variable_scope('q_fn', reuse=True):
            q_t, qt_list, _ = self._mlp_online(state_t, self.layers_online)
        # q_t = tf.stack(qt_list, axis=-1)

        # bayesian_loss = kl_loss
        bayesian_loss = 0.0
        for i in range(len(self.hparams.dense_layers)):
            bayesian_loss += self.layers_online['dense_{}'.format(i)].get_bayesian_loss()
        bayesian_loss += self.layers_online['dense_final'].get_bayesian_loss()

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

        # self._build_mlp(scope='q_tp1')
        # Target Network - calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            if self.hparams.target_type == 'Sample':
                _, q_tp1_list, q_tp1 = zip(
                    *[self._mlp_online(s_tp1, self.layers_target, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            elif self.hparams.target_type == 'MAP':
                q_tp1, _, _ = zip(
                    *[self._mlp_online(s_tp1, self.layers_target, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            else:
                _, _, q_tp1 = zip(
                    *[self._mlp_online(s_tp1, self.layers_target, reuse=tf.AUTO_REUSE) for s_tp1 in
                      state_tp1])
            # q_tp1 = [tf.stack(q_tp1_i, axis=-1) for q_tp1_i in q_tp1_list]
        # Target network parameters
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')
        print(q_tp1_vars)

        if self.double_q:
            with tf.variable_scope('q_fn', reuse=True):
                if self.hparams.doubleq_type == 'Sample':
                    _, _, q_tp1_online = zip(
                        *[self._mlp_online(s_tp1, self.layers_online, reuse=True) for s_tp1 in
                          state_tp1])
                elif self.hparams.doubleq_type == 'MAP':
                    if self.hparams.thomp_freq == 20:
                        q_tp1_online, q_tp1_online_list, _ = zip(*[self._mlp_online(s_tp1, self.layers_thomp,
                                                                                   reuse=True) for s_tp1 in state_tp1])
                    else:
                        q_tp1_online, q_tp1_online_list, _ = zip(*[self._mlp_online(s_tp1, self.layers_online,
                                                                                    reuse=True) for s_tp1 in state_tp1])

                # q_tp1_online = [tf.stack(q_tp1_online_i, axis=-1) for q_tp1_online_i in q_tp1_online_list]
            if self.num_bootstrap_heads:
                num_heads = self.num_bootstrap_heads
            else:
                num_heads = 1
            # determine the action to choose based on online Q estimator.
            q_tp1_online_idx = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for q in
                                q_tp1_online]
            # use the index from above to compute the value using the Target Network values q_tp1
            v_tp1 = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1, q_tp1_online_idx)], axis=0)
        else:
            v_tp1 = tf.stack([tf.reduce_max(q) for q in q_tp1], axis=0)

        # if s_{t+1} is the terminal state, we do not evaluate the Q value of
        # the state.
        q_tp1_masked = (1.0 - done_mask) * v_tp1
        q_t_target = reward_t + self.gamma * q_tp1_masked

        # stop gradient from flowing to the computational graph which computes
        # the Q value of s_{t+1}. (Target network parameters are not updated)
        # td_error has shape [batch_size, 1]
        td_target = tf.stop_gradient(q_t_target)
        td_error = td_target - q_t

        # Bayesian Loss (Expected variational log posterior - Expected log prior)
        weighted_bayesian_loss = bayesian_loss * kl_weight

        # Expected Log-likelihood Huber Loss
        errors = tf.where(tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5, 1.0 * (tf.abs(td_error) - 0.5))
        log_weighted_error = error_weight * errors
        weighted_error = tf.reduce_mean(log_weighted_error)  # reduce mean converts loss to scalar

        # Total Loss
        # total_loss = tf.reduce_mean(weighted_bayesian_loss + weighted_error)  # reduce mean converts loss to scalar
        total_loss = weighted_bayesian_loss + weighted_error  # reduce mean converts loss to scalar

        if self.built_number >= 2:
            self.built_flag1 = True
        if self.built_number == 4:
            self.built_flag2 = True
        print('Built status 1: ' + str(self.built_flag1))
        print('Built status 2: ' + str(self.built_flag2))

        # Calculate the mean gradient
        tvars = q_fn_vars
        # grads = [tf.squeeze(tf.clip_by_norm(tf.gradients(td_error, var), self.hparams.grad_clipping), axis=0) for var in tvars]
        #gradients = [
        #    tf.squeeze(tf.clip_by_norm(tf.gradients(td_error[:, :, i], tvars[0]), self.hparams.grad_clipping), 0) for i
        #    in range(10)]
        #mean_grad = tf.reduce_mean(tf.stack(gradients, axis=-1), axis=-1)
        #squared_differences = [tf.square(gradients[i] - mean_grad) for i in range(10)]
        #grad_var = tf.reduce_mean(tf.stack(squared_differences, axis=-1))

        # Log-likelihood variance
        #mu_td = tf.reduce_mean(td_error)
        #mu_td_centr = tf.square(td_error - tf.expand_dims(mu_td, axis=-1))
        #log_like_var = tf.reduce_mean(mu_td_centr)

        if self.hparams.thomp_freq == 20:
            return (q_values, td_error, weighted_error, weighted_bayesian_loss,
                    total_loss, q_fn_vars, q_tp1_vars, q_thomp_vars, grad_var, log_like_var)
        else:
            return (q_values, td_error, weighted_error, weighted_bayesian_loss,
                    total_loss, q_fn_vars, q_tp1_vars) #, grad_var, log_like_var)

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
            if self.hparams.thomp_freq == 20:
                (self.q_values,
                 self.td_error,
                 self.weighted_error,
                 self.weighted_bayesian_loss,
                 self.total_loss,
                 self.q_fn_vars,
                 self.q_tp1_vars,
                 self.q_thomp_vars,
                 self.grad_var,
                 self.log_like_var) = self._build_single_q_network(self.observations,
                                                                   self.head,
                                                                   self.state_t,
                                                                   self.state_tp1,
                                                                   self.done_mask,
                                                                   self.reward_t,
                                                                   self.error_weight,
                                                                   self.kl_weight)
            else:
                (self.q_values,
                 self.td_error,
                 self.weighted_error,
                 self.weighted_bayesian_loss,
                 self.total_loss,
                 self.q_fn_vars,
                 self.q_tp1_vars) = self._build_single_q_network(self.observations,
                                                                   self.head,
                                                                   self.state_t,
                                                                   self.state_tp1,
                                                                   self.done_mask,
                                                                   self.reward_t,
                                                                   self.error_weight,
                                                                   self.kl_weight)
                                                                   #self.episode,
                                                                   #self.sn)

            print('single q network built.' + '\n')
            self.action = self._action_train(self.q_values)

    def _action_train(self, q_vals):  # add a mode here?
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
            if self.hparams.opt == 'optimize_loss':
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
            # This alternative optimization gives us the grads directly so we can calculate its variance
            elif self.hparams.opt == 'apply_grads' and self.hparams.local_reparam is True:
                tvars = self.q_fn_vars
                grads = [
                    tf.squeeze(tf.clip_by_norm(tf.gradients(self.total_loss, var), self.hparams.grad_clipping), axis=0)
                    for var in tvars]
                learning_rate_decay_fn = functools.partial(tf.train.exponential_decay,
                                                           decay_steps=self.learning_rate_decay_steps,
                                                           decay_rate=self.learning_rate_decay_rate)  # change variables here
                self.learning_rate = learning_rate_decay_fn(self.learning_rate, tf.train.get_or_create_global_step())
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimization_op = optimizer.apply_gradients(zip(grads, tvars),
                                                                 global_step=tf.train.get_or_create_global_step(),
                                                                 name='train_step')

            # Updates the target network with new parameters every 20 episodes (update frequency)
            self.update_op = []
            for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                   sorted(self.q_tp1_vars, key=lambda v: v.name)):
                self.update_op.append(target.assign(var))
            self.update_op = tf.group(*self.update_op)

            if self.hparams.thomp_freq == 20:
                self.update_op2 = []
                for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                       sorted(self.q_thomp_vars, key=lambda v: v.name)):
                    self.update_op2.append(target.assign(var))
                self.update_op2 = tf.group(*self.update_op2)

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
                # self.weighted_error_summary = tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(self.weighted_error)))
                self.weighted_error_summary = tf.summary.scalar('td_error', tf.reduce_mean(self.weighted_error))
                # self.log_like_var_summary = tf.summary.scalar('log_likelihood_variance',
                                                              #tf.reduce_mean(tf.abs(self.log_like_var)))
                # self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                #                                                tf.reduce_mean(tf.abs(self.weighted_bayesian_loss)))
                self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                                                               tf.reduce_mean(self.weighted_bayesian_loss))
                # self.grad_var_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(tf.abs(self.grad_var)))
                # self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(tf.abs(self.total_loss)))
                self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(self.total_loss))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                self.episode_summary = tf.summary.merge([smiles_summary, reward_summary])

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

    def _run_action_op(self, observations, head): # add sn if changing TS freq.
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
        action = np.asscalar(tf.get_default_session().run(self.action, feed_dict={self.observations: observations,
                                                                                  self.head: head})) # add self.sn: sn if changing TS freq.
        return action

    def get_action(self,
                   observations,
                   stochastic=True,
                   head=0,
                   episode=None,
                   update_epsilon=None): # sn=None if changing TS freq.
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
            return action
        else:
            return self._run_action_op(observations, head) # sn

    def train(self, states, rewards, next_states, done, weight, kl_weight, ep, summary=True):
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
            if self.hparams.multi_obj:
                ops = [self.weighted_error, self.weighted_bayesian_loss, self.weighted_error_summary,
                       self.bayesian_loss_summary,
                       self.total_loss, self.total_loss_summary, self.grad_var_summary, self.log_like_var_summary,
                       self.optimization_op]
            else:
                ops = [self.td_error, self.weighted_error, self.weighted_bayesian_loss, self.weighted_error_summary,
                       self.bayesian_loss_summary,
                       self.total_loss, self.total_loss_summary, self.optimization_op] # self.grad_var, self.grad_var_summary,
                       # self.log_like_var_summary, self.optimization_op]
            # ops = [self.td_error, self.error_summary, self.optimization_op]
        else:
            ops = [self.td_error, self, total_loss, self.optimization_op]
        feed_dict = {self.state_t: states,
                     self.reward_t: rewards,
                     self.done_mask: done,
                     self.error_weight: weight,
                     self.kl_weight: kl_weight,
                     self.episode: ep}
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

    def _build_mlp(self, scope=None):
        # Problem with this one is: the first scope, q_fn_vars is perfect, but not the 2nd one
        # Defining the MLP (for when DenseReparam is used)
        # Online
        # self.sigma_prior, self.mixture_weights = self.mixture_gen(self.hparams.num_mixtures)
        if self.hparams.prior_type == 'mixed':
            sp = (float(np.exp(-0.5, dtype=np.float32)),  # -1.0 (original)
                  float(np.exp(-2.0, dtype=np.float32)))
            weight_prior = WeightPriorMOG(sigma_prior=sp,  # Should be list, even if just 1 value.
                                          mixture_weights=(0.25, 0.75),
                                          calculation_type=self.hparams.bayesian_loss)
        elif self.hparams.prior_type == 'single':
            sp = (float(np.exp(-1.373835783, dtype=np.float32)),)
            weight_prior = WeightPriorMOG(sigma_prior=sp,  # Should be list, even if just 1 value.
                                          mixture_weights=(1.0, ),
                                          calculation_type=self.hparams.bayesian_loss)
        elif self.hparams.prior_type == 'ard':
            weight_prior = WeightPriorARD()
        elif self.hparams.prior_type == 'student':
            weight_prior = WeightPriorStudent()
        else:
            raise ValueError("Invalid prior type.")
        if scope == 'q_fn':
            self.layers_online = dict()
            print('I am building the online mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_online['dense_{}'.format(i)] = DenseReparameterisation(units=self.units[i],
                                                                                   weight_prior=weight_prior,
                                                                                   variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                                   use_clt=self.hparams.use_clt,
                                                                                   activation=None,
                                                                                   uncertainty_propagation_type=None,
                                                                                   trainable=True,
                                                                                   use_bias=self.hparams.use_bias,
                                                                                   name=scope + 'dense_{}'.format(i),
                                                                                   reuse=True)
            self.layers_online['dense_final'] = DenseReparameterisation(units=self.units[-1],
                                                                        weight_prior=weight_prior,
                                                                        variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                        use_clt=self.hparams.use_clt,
                                                                        activation=None,
                                                                        uncertainty_propagation_type=None,
                                                                        trainable=True,
                                                                        use_bias=self.hparams.use_bias,
                                                                        name=scope + 'dense_final',
                                                                        reuse=True)
            self.built_number += 1
        # Target
        elif scope == 'q_tp1':
            self.layers_target = dict()
            print('I am building the target mlp ... \n')
            for i in range(len(self.hparams.dense_layers)):
                self.layers_target['dense_{}'.format(i)] = DenseReparameterisation(units=self.units[i],
                                                                                   weight_prior=weight_prior,
                                                                                   variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                                   use_clt=self.hparams.use_clt,
                                                                                   activation=None,
                                                                                   uncertainty_propagation_type=None,
                                                                                   trainable=True,
                                                                                   use_bias=self.hparams.use_bias,
                                                                                   name=scope + 'dense_{}'.format(i),
                                                                                   reuse=True)
            self.layers_target['dense_final'] = DenseReparameterisation(units=self.units[-1],
                                                                        weight_prior=weight_prior,
                                                                        variance_parameterisation_type=self.hparams.variance_parameterisation_type,
                                                                        use_clt=self.hparams.use_clt,
                                                                        activation=None,
                                                                        uncertainty_propagation_type=None,
                                                                        trainable=True,
                                                                        use_bias=self.hparams.use_bias,
                                                                        name=scope + 'dense_final',
                                                                        reuse=True)

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
            self.reward_t = tf.placeholder(tf.float32, (batch_size, self.num_objectives), name='reward_t')
            # objective_weight is the weight to scalarize the objective vector:
            # reward = sum (objective_weight_i * objective_i)
            self.objective_weight_input = tf.placeholder(tf.float32, [self.num_objectives, 1], name='objective_weight')

            # split reward for each q network
            rewards_list = tf.split(self.reward_t, self.num_objectives, axis=1)
            q_values_list = []
            self.td_error = []
            self.weighted_error = 0
            self.weighted_bayesian_loss = 0
            self.grad_var = 0
            self.log_like_var = 0
            self.total_loss = 0
            self.q_fn_vars = []
            self.q_tp1_vars = []

            # build a Q network for each objective
            for obj_idx in range(self.num_objectives):
                with tf.variable_scope('objective_%i' % obj_idx):
                    (q_values,
                     td_error,
                     weighted_error,
                     weighted_bayesian_loss,
                     total_loss,
                     q_fn_vars,
                     q_tp1_vars,
                     grad_var,
                     log_like_var) = self._build_single_q_network(self.observations,
                                                                  self.head,
                                                                  self.state_t,
                                                                  self.state_tp1,
                                                                  self.done_mask,
                                                                  rewards_list[obj_idx],
                                                                  self.error_weight,
                                                                  self.kl_weight,
                                                                  self.episode)
                    q_values_list.append(tf.expand_dims(q_values, 1))
                    # td error is for summary only.
                    # weighted error is the optimization goal.
                    self.td_error.append(td_error)
                    self.weighted_error += weighted_error / self.num_objectives
                    self.weighted_bayesian_loss += weighted_bayesian_loss / self.num_objectives
                    self.total_loss += total_loss / self.num_objectives
                    self.grad_var += grad_var / self.num_objectives
                    self.log_like_var += log_like_var / self.num_objectives
                    self.q_fn_vars += q_fn_vars
                    self.q_tp1_vars += q_tp1_vars

                print('single q network {} built.'.format(obj_idx) + '\n')

            self.q_values = tf.concat(q_values_list, axis=1)
            # action is the one that leads to the maximum weighted reward.
            self.action = tf.argmax(tf.matmul(self.q_values, self.objective_weight_input), axis=0)

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
                # error_summaries = [tf.summary.scalar('td_error_%i' % i, tf.reduce_mean(tf.abs(self.td_error[i]))) for i
                #                    in range(self.num_objectives)]
                error_summaries = [tf.summary.scalar('td_error_%i' % i, tf.reduce_mean(self.td_error[i])) for i
                                   in range(self.num_objectives)]
                self.weighted_error_summary = tf.summary.merge(error_summaries)
                # self.log_like_var_summary = tf.summary.scalar('log_likelihood_variance',
                #                                               tf.reduce_mean(tf.abs(self.log_like_var)))
                self.log_like_var_summary = tf.summary.scalar('log_likelihood_variance',
                                                              tf.reduce_mean(self.log_like_var))
                # self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                #                                                tf.reduce_mean(tf.abs(self.weighted_bayesian_loss)))
                self.bayesian_loss_summary = tf.summary.scalar('bayesian_loss',
                                                               tf.reduce_mean(self.weighted_bayesian_loss))
                # self.grad_var_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(tf.abs(self.grad_var)))
                self.grad_var_summary = tf.summary.scalar('gradient_variance', tf.reduce_mean(self.grad_var))
                # self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(tf.abs(self.total_loss)))
                self.total_loss_summary = tf.summary.scalar('total_loss', tf.reduce_mean(self.total_loss))

                # SMILES and Weighted Reward Summaries
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.rewards = [tf.placeholder(tf.float32, [], 'summary_reward_obj_%i' % i) for i in
                                range(self.num_objectives)]
                # Weighted sum of the rewards.
                self.weighted_reward = tf.placeholder(tf.float32, [], 'summary_reward_sum')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summaries = [tf.summary.scalar('reward_obj_%i' % i, self.rewards[i]) for i in
                                    range(self.num_objectives)]
                # reward_summaries.append(tf.summary.scalar('sum_reward', self.rewards[-1]))
                reward_summaries.append(tf.summary.scalar('sum_reward', self.weighted_reward))
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


def get_hparams(**kwargs):
    """Get the hyperparameters for the model from a json object.

  Returns:
    A HParams object containing all the hyperparameters.
  """
    hparams = contrib_training.HParams(
        atom_types=['C', 'O', 'N', 'F'],
        max_steps_per_episode=40,
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=False,
        allowed_ring_sizes=[3, 4, 5, 6, 7],  # [5,6]
        replay_buffer_size=5000,  # 1000000, 5000 (default)
        learning_rate=1e-4,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.9,  # 0.8
        num_episodes=5000,  # 5000 for optimize_qed
        batch_size=16,  # 64, 12 (default)
        learning_frequency=4,
        update_frequency=20,
        thomp_freq=1,  # 20
        grad_clipping=10.0,
        gamma=0.9,
        double_q=True,
        num_bootstrap_heads=0,  # 12
        prioritized=False,
        prioritized_alpha=0.6,  # 0.6, Try with 1.0
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
        prior_type='ard',  # mixed, single (CF known), student, ard (CF known)
        bayesian_loss='MC',  # MC or Closed (refers to MoG prior only)
        sigma_prior=(float(np.exp(-1.0, dtype=np.float32)), float(np.exp(-2.0, dtype=np.float32))),
        var_range=[-4.6, -3.9],
        target_type='Sample',  # Sample
        doubleq_type='Sample',  # MAP
        use_bias=True,
        use_clt=True,
        n_samples=1,
        variance_parameterisation_type='layer_wise',  # layer_wise, neuron_wise, weight_wise, additive
        opt='optimize_loss',
        multi_obj=False,
        rbs=False,
        uq='stochastic')  # stochastic generally beter than closed_form, but less efficient
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
