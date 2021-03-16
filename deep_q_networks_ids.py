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
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import training as contrib_training

from bbb_utils import *



class DeepQNetworkIDS(object):
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
                 q_fn,
                 learning_rate=0.001,
                 learning_rate_decay_steps=10000,
                 learning_rate_decay_rate=0.8,
                 optimizer='Adam',
                 grad_clipping=None,
                 gamma=1.0,
                 epsilon=0.2,
                 double_q=True,
                 dueling=False,
                 num_bootstrap_heads=12,
                 scope='bdqn',
                 reuse=None,
                 exploration='',
                 num_sigma=2):
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
        self.ipt_shape = ipt_shape
        self.q_fn = q_fn
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
        self.loss = 'huber'
        self.exploration = exploration
        self.num_sigma = num_sigma
        self.hparams = hparams

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
            # self.obs = tf.placeholder(tf.float32, [None, fingerprint_length], name='obs')
            # head is the index of the head we want to choose for decison.
            # See https://arxiv.org/abs/1703.07608
            self.head = tf.placeholder(tf.int32, [], name='head')
            self.aleatoric_head = tf.placeholder(tf.int32, [], name='aleatoric_head')
            # When sample from memory, the batch_size can be fixed, as it is
            # possible to sample any number of samples from memory.
            # state_t is the state at time step t
            self.state_t = tf.placeholder(tf.float32, self.ipt_shape, name='state_t')
            # state_tp1 is the state at time step t + 1, tp1 is short for t plus 1.
            self.state_tp1 = [tf.placeholder(tf.float32, [None, fingerprint_length], name='state_tp1_%i' % i) for i in
                              range(batch_size)]
            # done_mask is a {0, 1} tensor indicating whether state_tp1 is the
            # terminal state.
            self.done_mask = tf.placeholder(tf.float32, (batch_size, 1), name='done_mask')
            self.error_weight = tf.placeholder(tf.float32, (batch_size, 1), name='error_weight')


    def _build_single_q_network(self, observations, head, aleatoric_head, state_t, state_tp1,
                                done_mask, reward_t, error_weight):
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
        with tf.variable_scope('q_fn'):
            # q_value have shape [batch_size, 1]. For one head.
            q_vals = self.q_fn(observations)
            q_values = tf.gather(q_vals, head, axis=-1)
            # q_values for all heads (to calculate epistemic uncertainty)
            q_values_all = tf.gather(q_vals, list(range(self.num_bootstrap_heads)), axis=-1)
            # aleatoric uncertainty
            log_pred_var = tf.gather(q_vals, aleatoric_head, axis=-1)

        # calculating q_fn(state_t)
        # The Q network shares parameters with the action graph.
        with tf.variable_scope('q_fn', reuse=True):
            qt = self.q_fn(state_t, reuse=True)
            q_t = tf.gather(qt, list(range(self.num_bootstrap_heads)), axis=-1)
            lpv = tf.expand_dims(tf.gather(qt, aleatoric_head, axis=-1), 1)
            # The average over heads:
            mu = tf.reduce_mean(q_t, axis=1)
            print(mu.shape)
            print(tf.reduce_max(mu, axis=-1, keepdims=True))
            # Centralise the mean (difference between mean and q-vals)
            mu_centralised = q_t - tf.expand_dims(mu, axis=1)
            # Average Epistemic Uncertainty for current step
            e_uncert = tf.reduce_mean(tf.square(mu_centralised), axis=1)

        # Online network parameters
        q_fn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn')

        # calculating q_fn(state_tp1)
        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            q_tp1 = [tf.gather(self.q_fn(s_tp1, reuse=tf.AUTO_REUSE), list(range(self.num_bootstrap_heads)), axis=-1)
                     for s_tp1 in state_tp1]
        # Target network parameters
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')

        if self.double_q:
            with tf.variable_scope('q_fn', reuse=True):
                q_tp1_online = [tf.gather(self.q_fn(s_tp1, reuse=True), list(range(self.num_bootstrap_heads)), axis=-1)
                                for s_tp1 in state_tp1]
            if self.num_bootstrap_heads:
                num_heads = self.num_bootstrap_heads
            else:
                num_heads = 1
            # determine the action to choose based on online Q estimator.
            q_tp1_online_idx = [tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for q in
                                q_tp1_online]
            # use the index from max online q_values to compute the value
            # function
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

        # If use bootstrap, each head is trained with a different subset of the
        # training sample. Like the idea of dropout.
        if self.num_bootstrap_heads:
            # This head mask is m_t in the original bootstrapped dqn paper (appendix B)
            # It decides which heads should be trained on the experience generated from the buffer at time step t
            head_mask = tf.keras.backend.random_binomial(shape=(1, self.num_bootstrap_heads), p=0.6)
            td_error = tf.reduce_mean(td_error * head_mask, axis=1)

        # Expected Log-likelihood Huber Loss
        errors = tf.where(tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5, 1.0 * (tf.abs(td_error) - 0.5))
        errors = (errors * tf.exp(-1 * lpv))
        weighted_error = tf.reduce_mean(error_weight * errors + 0.5 * lpv)

        return q_values, q_values_all, q_t, td_error, weighted_error, q_fn_vars, q_tp1_vars, log_pred_var, lpv, e_uncert

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
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            self.reward_t = tf.placeholder(tf.float32, (batch_size, 1), name='reward_t')
            # The Q network shares parameters with the action graph.
            # tenors start with q or v have shape [batch_size, 1] when not using
            # bootstrap. When using bootstrap, the shapes are
            # [batch_size, num_bootstrap_heads]
            (self.q_values,
             self.q_values_all,
             self.q_t,
             self.td_error,
             self.weighted_error,
             self.q_fn_vars,
             self.q_tp1_vars,
             self.log_pred_var,
             self.aleatoric_uncertainty,
             self.epistemic_uncertainty) = self._build_single_q_network(self.observations,
                                                                        # self.obs,
                                                                        self.head,
                                                                        self.aleatoric_head,
                                                                        self.state_t,
                                                                        self.state_tp1,
                                                                        self.done_mask,
                                                                        self.reward_t,
                                                                        self.error_weight)

            self.action_buffer = self._action_train(self.q_values)
            self.action = self._action_train_IDS(self.q_values_all,
                                                 self.log_pred_var)

            (self.ids_value, self.ig, self.e_uncert) = self.ids(self.q_values_all, self.log_pred_var,
                                                                None)

    def _action_train(self, q_vals):
        """Defines the action selection policy during training.

        :param: q_vals:
            The q values of the observations

        """
        return tf.argmax(q_vals)

    def _action_train_IDS(self, q_vals, log_pred_var):
        """Defines the action selection during training.
        Based on the IDS Exploration method,
        code inspired by https://github.com/nikonikolov/rltf/blob/master/rltf/models/dqn_ids.py


        """
        self.eps = 1e-8
        # Return distribution variance
        self.rho2 = tf.exp(log_pred_var)
        self.sum_rho2 = tf.reduce_sum(self.rho2)
        self.no_actions = self.rho2.shape[0].value

        # Mean (averaging over heads)
        self.mu = tf.reduce_mean(q_vals, axis=1)

        # Centralise the mean (difference between mean and q-vals)
        self.mu_centralised = q_vals - tf.expand_dims(self.mu, axis=-1)

        # Variance and Standard Deviation
        self.var = tf.reduce_mean(tf.square(self.mu_centralised), axis=1)
        self.std = tf.sqrt(self.var)

        # Compute the Estimated Regret
        self.regret = tf.reduce_max(self.mu + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                    self.mu - self.num_sigma * self.std)

        # Compute the normalised aleatoric variance
        if self.no_actions is not None:
            self.rho2_norm = self.rho2 / (self.eps + (1 / self.no_actions) * self.sum_rho2)
        else:
            self.rho2_norm = self.rho2

        # Compute the Information Gain
        self.inf_gain = tf.log(1 + self.var / self.rho2_norm) + self.eps

        # Compute Regret-Information ratio
        self.ids_score = tf.square(self.regret) / self.inf_gain

        # Get action
        a = tf.argmin(self.ids_score, axis=-1)

        # Histograms for debugging and visualisation
        # self.mean_hist = tf.summary.histogram("debug/mean", self.mu)
        # self.std_hist = tf.summary.histogram("debug/std", self.std)
        # self.regret_hist = tf.summary.histogram("debug/regret", self.regret)
        # self.inf_gain_hist = tf.summary.histogram("debug/inf_gain", self.inf_gain)
        # self.ids_hist = tf.summary.histogram("debug/ids", self.ids_score)

        return a

    def ids(self, q_vals, log_pred_var, e_closed, e_uncert_norm=None, e_uncert2=None, e_uncert2_norm=None, w_list=None,
            b_list=None, w=None, b=None):
        """Defines the action selection during training.
        Based on the IDS Exploration method,
        code inspired by https://github.com/nikonikolov/rltf/blob/master/rltf/models/dqn_ids.py

        """
        self.eps = 1e-9
        # Return distribution variance (aleatoric variance)
        self.rho2 = tf.exp(log_pred_var)
        self.sum_rho2 = tf.reduce_sum(self.rho2)
        self.no_actions = self.rho2.shape[0].value

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

        # Compute the Estimated Regret
        self.regret = tf.reduce_max(self.mu + self.num_sigma * self.std, axis=-1, keepdims=True) - (
                self.mu - self.num_sigma * self.std)

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

        return (ids_score, self.inf_gain, self.var)

    def get_ids_score(self, state_t, head, aleatoric_head):
        # if state_t.shape[0] != self.hparams.batch_size:
        ids = tf.get_default_session().run(self.ids_value, feed_dict={self.observations: state_t,
                                                                      self.head: head,
                                                                      self.aleatoric_head: aleatoric_head})
        inf_gain = tf.get_default_session().run(self.ig, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head})
        e1 = tf.get_default_session().run(self.e_uncert, feed_dict={self.observations: state_t,
                                                                    self.head: head,
                                                                    self.aleatoric_head: aleatoric_head})
        return ids, inf_gain, e1

    def _build_training_ops(self):
        """Creates the training operations.

    Instance attributes created:
      optimization_op: the operation of optimize the loss.
      update_op: the operation to update the q network.
    """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.optimization_op = contrib_layers.optimize_loss(
                loss=self.weighted_error,
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
                self.qt_summary = tf.summary.scalar('qt', tf.reduce_mean(tf.reduce_mean(self.q_t, axis=1), axis=0))
                self.error_summary = tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(self.td_error)))
                self.aleatoric_summary = tf.summary.scalar('aleatoric_uncertainty',
                                                           tf.reduce_mean(tf.abs(self.aleatoric_uncertainty)))
                self.epistemic_summary = tf.summary.scalar('epistemic_uncertainty',
                                                           tf.reduce_mean(tf.abs(self.epistemic_uncertainty)))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                self.reward_adjusted = tf.placeholder(tf.float32, [], 'summary_reward_adjusted')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                reward_summary_adjusted = tf.summary.scalar('reward', self.reward_adjusted)
                self.episode_summary = tf.summary.merge([smiles_summary, reward_summary])
                self.episode_summary_adjusted = tf.summary.merge([smiles_summary, reward_summary_adjusted])

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

    def get_q_vals(self, obs):

        return tf.get_default_session().run(self.q_values_all,
                                            feed_dict={self.obs: obs})

    def _run_action_op(self, observations, head, aleatoric_head):
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
        return np.asscalar(tf.get_default_session().run(self.action_buffer, feed_dict={self.observations: observations,
                                                                                       self.head: head,
                                                                                       self.aleatoric_head: aleatoric_head}))

    def get_ids_action(self, obs, aleatoric_head, update_epsilon=None):
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        action = np.asscalar(tf.get_default_session().run(self.action, feed_dict={self.observations: obs,
                                                                                  self.aleatoric_head: aleatoric_head}))

        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, obs.shape[0])
        else:
            return action

    def get_action(self,
                   observations,
                   stochastic=True,
                   head=0,
                   aleatoric_head=None,
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
            return self._run_action_op(observations, head, aleatoric_head)

    def train(self, head, aleatoric_head, states, rewards, next_states, done, weight,
              summary=True):  # add kl_weight later
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
            ops = [self.td_error, self.qt_summary, self.error_summary, self.aleatoric_uncertainty,
                   self.aleatoric_summary, self.epistemic_uncertainty, self.epistemic_summary,
                   self.q_values, self.q_values_all, self.optimization_op]
        else:
            ops = [self.td_error, self.optimization_op]
        feed_dict = {self.head: head,
                     self.aleatoric_head: aleatoric_head,
                     # self.obs: states,
                     self.state_t: states,
                     self.observations: states,
                     self.reward_t: rewards,
                     self.done_mask: done,
                     self.error_weight: weight}
        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state
        return tf.get_default_session().run(ops, feed_dict=feed_dict)


class MultiObjectiveDeepQNetworkIDS(DeepQNetworkIDS):
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


def multi_layer_model_ids(inputs, hparams, reuse=None):
    """Multi-layer model for q learning.

  Args:
    inputs: Tensor. The input.
    hparams: tf.HParameters. The hyper-parameters.
    reuse: Boolean. Whether the parameters should be reused.

  Returns:
    Tensor. shape = [batch_size, hparams.num_bootstrap_heads]. The output.
  """
    output = inputs
    for i, units in enumerate(hparams.dense_layers):
        output = tf.layers.dense(output, units, name='dense_%i' % i, reuse=reuse)
        output = getattr(tf.nn, hparams.activation)(output)
        if hparams.batch_norm:
            output = tf.layers.batch_normalization(output, fused=True, name='bn_%i' % i, reuse=reuse)
    if hparams.num_bootstrap_heads:
        output_dim = hparams.num_bootstrap_heads + 1
    else:
        output_dim = 1
    output = tf.layers.dense(output, output_dim, name='final', reuse=reuse)
    return output


def multi_layer_model(inputs, hparams, reuse=None):
    """Multi-layer model for q learning.

  Args:
    inputs: Tensor. The input.
    hparams: tf.HParameters. The hyper-parameters.
    reuse: Boolean. Whether the parameters should be reused.

  Returns:
    Tensor. shape = [batch_size, hparams.num_bootstrap_heads]. The output.
  """
    output = inputs
    for i, units in enumerate(hparams.dense_layers):
        output = tf.layers.dense(output, units, name='dense_%i' % i, reuse=reuse)
        output = getattr(tf.nn, hparams.activation)(output)
        if hparams.batch_norm:
            output = tf.layers.batch_normalization(output, fused=True, name='bn_%i' % i, reuse=reuse)
    output_dim = hparams.num_bootstrap_heads + 1
    output = tf.layers.dense(output, output_dim, name='final', reuse=reuse)
    return output


def get_hparams(**kwargs):
    """Get the hyperparameters for the model from a json object.

  Args:
    **kwargs: Dict of parameter overrides.
  Possible keyword arguments:
    atom_types: Dict. The possible atom types in the molecule.
    max_steps_per_episode: Integer. The maximum number of steps for one episode.
    allow_removal: Boolean. Whether to allow removal of a bond.
    allow_no_modification: Boolean. If true, the valid action set will include
      doing nothing to the current molecule, i.e., the current molecule itself
      will be added to the action set.
    replay_buffer_size: Integer. The size of the replay buffer.
    learning_rate: Float. Learning rate.
    learning_rate_decay_steps: Integer. The number of steps between each
      learning rate decay.
    learning_rate_decay_rate: Float. The rate of learning rate decay.
    num_episodes: Integer. Number of episodes to run.
    batch_size: Integer. The batch size.
    learning_frequency: Integer. The number of steps between each training
      operation.
    update_frequency: Integer. The number of steps between each update of the
      target Q network
    grad_clipping: Integer. maximum value of the gradient norm.
    gamma: Float. The discount factor for the reward.
    double_q: Boolean. Whether to used double Q learning.
      See https://arxiv.org/abs/1509.06461 for detail.
    bootstrap: Integer. The number of bootstrap heads. See
      https://arxiv.org/abs/1703.07608 for detail.
    prioritized: Boolean. Whether to use prioritized replay. See
      https://arxiv.org/abs/1511.05952 for detail.
    prioritized_alpha: Float. The parameter alpha in the prioritized replay.
    prioritized_beta: Float. The parameter beta in the prioritized replay.
    prioritized_epsilon: Float. The parameter epsilon in the prioritized replay.
    fingerprint_radius: Integer. The radius of the Morgan fingerprint.
    fingerprint_length: Integer. The length of the Morgan fingerprint.
    dense_layers: List of integers. The hidden units in the dense layers.
    activation: String. The activation function to use.
    optimizer: String. The optimizer to use.
    batch_norm: Boolean. Whether to use batch normalization.
    save_frequency: Integer. The number of episodes between each saving.

  Returns:
    A HParams object containing all the hyperparameters.
  """
    hparams = contrib_training.HParams(
        atom_types=['C', 'O', 'N'],
        max_steps_per_episode=40,
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=False,
        allowed_ring_sizes=[5, 6],
        replay_buffer_size=5000,
        learning_rate=1e-4,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.9,  # 0.8
        num_episodes=5000,
        batch_size=128,  # 64
        kl_weighting='uniform',
        learning_frequency=4,
        update_frequency=20,
        grad_clipping=10.0,
        gamma=0.9,
        double_q=True,
        num_bootstrap_heads=12,
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
        exp='ids', # thompson or ids
        reward_eng='ids',
        norm=False,
        re=True,
        multi_obj=False)
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