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

"""Executor for deep Q network models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from baselines.common import schedules
from baselines.deepq import replay_buffer

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import b_deep_q_networks_ids_refactored
import molecules_ids as molecules_mdp
import molecules_py
import core
from store_best_molecules import MoleculeMonitor

flags.DEFINE_string('model_dir',
                    '/vol/bitbucket/pds14/bdqn_ids/56lr3SS10_uf10_bs16',
                    'The directory to save data to.')
flags.DEFINE_string('target_molecule', 'C1CCC2CCCCC2C1',
                    'The SMILES string of the target molecule.')
flags.DEFINE_string('start_molecule', None,
                    'The SMILES string of the start molecule.')
flags.DEFINE_float(
    'similarity_weight', 0.5,
    'The weight of the similarity score in the reward function.')
flags.DEFINE_float('target_weight', 493.60,
                   'The target molecular weight of the molecule.')
flags.DEFINE_string('hparams', None, 'Filename for serialized HParams.')
flags.DEFINE_boolean('multi_objective', False,
                     'Whether to run multi objective DQN.')

FLAGS = flags.FLAGS

# This is the environment
class TargetWeightMolecule(molecules_mdp.Molecule):
    """Defines the subclass of a molecule MDP with a target molecular weight."""

    def __init__(self, target_weight, **kwargs):
        """Initializes the class.

            Args:
              target_weight: Float. the target molecular weight.
              **kwargs: The keyword arguments passed to the parent class.
            """
        super(TargetWeightMolecule, self).__init__(**kwargs)
        self.target_weight = target_weight

    def _reward(self):
        """Calculates the reward of the current state.

            The reward is defined as the negative l1 distance between the current
            molecular weight and target molecular weight range.

            Returns:
              Float. The negative distance.
            """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return -self.target_weight ** 2
        mw = Descriptors.MolWt(molecule)
        if lower <= mw <= upper:
            return 1
        return -min(abs(lower - mw), abs(upper - mw))


class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


############################################ THIS CODE IS ORIGINAL ###########################################
# Inspired by code from: https://github.com/hill-a/stable-baselines
class KLSchedule(Schedule):
    """
    Defines the Schedule for weighting the KL Divergence term in the loss

    :param M: (float)
        The number of mini-batches used for an update-op
    :param _outside_value: (float)
         if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested..
    """

    def __init__(self, hparams, ep, weighting_type, outside_value=None):
        self._max_steps_per_episode = hparams.max_steps_per_episode
        self._update_frequency = hparams.update_frequency
        # self.M = float(self._max_steps_per_episode / self._update_frequency * hparams.num_episodes)
        self.M = float(hparams.num_episodes)
        self._outside_value = outside_value
        self.weighting_type = weighting_type
        self.episode = ep

    def value(self, step):
        if self.weighting_type is None:
            return np.array([1e-5])
        elif self.weighting_type == 'null':
            return np.array([0.0])
        elif self.weighting_type == 'uniform':
            return np.array([1.0 / self.M])
        elif self.weighting_type == 'soenderby':
            return np.array([min(self.episode / (hparams.num_episodes // 4), 1)])
        elif self.weighting_type == 'blundell':
            return np.array([(2 ** (self.M - step)) / (2 ** self.M - 1)])

        assert self._outside_value is not None
        return self._outside_value



class MultiObjectiveRewardMolecule(molecules_mdp.Molecule):
    """Defines the subclass of generating a molecule with a specific reward.

      The reward is defined as a 1-D vector with 2 entries: similarity and QED
        reward = (similarity_score, qed_score)
      """

    def __init__(self, target_molecule, **kwargs):
        """Initializes the class.

            Args:
              target_molecule: SMILES string. the target molecule against which we
                calculate the similarity.
              **kwargs: The keyword arguments passed to the parent class.
            """
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._target_mol_scaffold = molecules_py.get_scaffold(target_molecule)
        self.reward_dim = 2

    def get_fingerprint(self, molecule):
        """Gets the morgan fingerprint of the target molecule.

            Args:
              molecule: Chem.Mol. The current molecule.

            Returns:
              rdkit.ExplicitBitVect. The fingerprint of the target.
            """
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.

            Args:
              smiles: String. The SMILES string for the current molecule.

            Returns:
              Float. The Tanimoto similarity.
            """

        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint, fingerprint_structure)

    def _reward(self):
        """Calculates the reward of the current state.

            The reward is defined as a tuple of the similarity and QED value.

            Returns:
              A tuple of the similarity and qed value
            """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0, 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0, 0.0
        if molecules_py.contains_scaffold(mol, self._target_mol_scaffold):
            similarity_score = self.get_similarity(self._state)
        else:
            similarity_score = 0.0
        # calculate QED
        qed_value = QED.qed(mol)
        return similarity_score, qed_value


# TODO(zzp): use the tf.estimator interface.
def run_training(hparams, environment, bdqn):
    """Runs the training procedure.

      Briefly, the agent runs the action network to get an action to take in
      the environment. The state transition and reward are stored in the memory.
      Periodically the agent samples a batch of samples from the memory to
      update(train) its Q network. Note that the Q network and the action network
      share the same set of parameters, so the action network is also updated by
      the samples of (state, action, next_state, reward) batches.


      Args:
        hparams: tf.contrib.training.HParams. The hyper parameters of the model.
        environment: molecules.Molecule. The environment to run on.
        bdqn: An instance of the Bayesian DeepQNetwork class.

      Returns:
        None
      """
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir)

    molecule_monitor = MoleculeMonitor(FLAGS.model_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:

        # Build the dqn
        bdqn.build()

        # Define a model saver
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)

        # The schedule for the epsilon in epsilon greedy policy.
        if hparams.exp == 'thompson':
            exploration = schedules.PiecewiseSchedule(
                [(0, 1.0), (int(hparams.num_episodes * 0.5), 0.1), (hparams.num_episodes, 0.01)], outside_value=0.01)
        else:
            exploration = schedules.PiecewiseSchedule(
                [(0, 1.0), (int(hparams.num_episodes * 0.5), 0.1), (hparams.num_episodes, 0.01)], outside_value=0.01)

        # Define the memory variable as Prioritised Experience Replay Buffer from baselines modules
        if hparams.prioritized:
            memory = replay_buffer.PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
            beta_schedule = schedules.LinearSchedule(hparams.num_episodes, initial_p=hparams.prioritized_beta,
                                                     final_p=0)
        else:
            memory = replay_buffer.ReplayBuffer(hparams.replay_buffer_size)
            beta_schedule = None

        # The schedule for the Bayesian Loss
        kl_schedule = schedules.LinearSchedule(hparams.num_episodes, 1e-6, initial_p=1e-5)

        # Run the session with the global variables initialiser
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

        # Update the dqn for the first time after initialising all variables
        sess.run(bdqn.update_op)

        # Run an episode
        global_step = 0
        for episode in range(hparams.num_episodes):
            global_step = _episode(
                environment=environment,
                bdqn=bdqn,
                memory=memory,
                episode=episode,
                global_step=global_step,
                hparams=hparams,
                summary_writer=summary_writer,
                molecule_monitor=molecule_monitor,
                exploration=exploration,
                beta_schedule=beta_schedule,
                # kl_weight_schedule=KLSchedule(hparams, ep=episode, weighting_type=hparams.weighting_type)
                kl_weight_schedule=kl_schedule)
            # Update the target network every 'hparams.update_frequency' episodes
            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(bdqn.update_op)
            # Save the model every 'hparams.save_frequency' episodes (currently set to every 10 updates)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(sess, os.path.join(FLAGS.model_dir, 'ckpt'), global_step=global_step)

        molecule_monitor.store_molecules()


def _episode(environment, bdqn, memory, episode, global_step, hparams, summary_writer, molecule_monitor, exploration, beta_schedule,
             kl_weight_schedule):
    """Runs a single episode.

      Args:
        environment: molecules.Molecule; the environment to run on.
        bdqn: Bayesian DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        global_step: Integer global step; the total number of steps across all
            episodes.
        hparams: HParams.
        summary_writer: FileWriter used for writing Summary protos.
        exploration: Schedule used for exploration in the environment.
        beta_schedule: Schedule used for prioritized replay buffers.
        kl_weight_schedule: Schedule used to decay KL loss, if defined outside of
            this function.

      Returns:
        Updated global_step.
      """
    episode_start_time = time.time()
    environment.initialize()
    head = 0
    aleatoric_head = int(1)
    sample_number = np.random.randint(0, hparams.n_samples)
    episode_sequence = []
    for step in range(hparams.max_steps_per_episode):
        # Takes a step
        result = _step_nheads(
            environment=environment,
            bdqn=bdqn,
            memory=memory,
            episode=episode,
            global_step=global_step,
            hparams=hparams,
            exploration=exploration,
            head=head,
            aleatoric_head=aleatoric_head,
            sample_number=sample_number,
            summary_writer=summary_writer)
        episode_sequence.append(result)

        molecule_monitor.add_molecule(state=result.state,
                                      reward=result.reward)

        if hparams.reward_eng:
            episode_summary, adjusted_episode_summary = bdqn.log_result(result.state, result.reward)
            summary_writer.add_summary(adjusted_episode_summary, global_step)
        else:
            episode_summary = bdqn.log_result(result.state, result.reward)
        summary_writer.add_summary(episode_summary, global_step)

        if step == hparams.max_steps_per_episode - 1:
            max_reward = 0.0
            best_state = None
            for e in episode_sequence:
                if e.reward > max_reward:
                    max_reward = e.reward
                    best_state = e.state
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes, time.time() - episode_start_time)
            logging.info('SMILES: %s', best_state)
            # Use %s since reward can be a tuple or a float number.
            # logging.info('IDS term: %s ', str(result.ids_term)) # Used for reward_eng only
            logging.info('The raw reward is: %s\n', str(max_reward))
            # logging.info('The adjusted reward is: %s\n', str(result.reward_adjusted))
        # Waits for the replay buffer to contain at least 50 * max_steps_per_episode t
        # transitions before sampling and updating the dqn
        if (episode > min(50, hparams.num_episodes / 10)) and (global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                (state_t, _, reward_t, state_tp1, done_mask, weight, indices) = memory.sample(hparams.batch_size,
                                                                                              beta=beta_schedule.value(
                                                                                                  episode))
            else:
                (state_t, _, reward_t, state_tp1, done_mask) = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
            # KL weighting
            kl_weight = kl_weight_schedule.value(episode)
            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
            episode_tensor = np.array([episode]).astype(float)
            if hparams.uq == 'closed':
                (td_error,
                 weighted_error,
                 predictive_mean_summary,
                 weighted_bayesian_loss,
                 weighted_error_summary,
                 bayesian_loss_summary,
                 aleatoric_uncertainty, aleatoric_summary, aleatoric_normalised_summary,
                 epistemic_uncertainty, epistemic_summary,
                 grad_var_summary,
                 log_like_var_summary,
                 total_loss,
                 total_loss_summary, _) = bdqn.train(states=state_t,
                                                     rewards=reward_t,
                                                     next_states=state_tp1,
                                                     done=np.expand_dims(done_mask, axis=1),
                                                     weight=np.expand_dims(weight, axis=1),
                                                     kl_weight=kl_weight,
                                                     ep=episode_tensor,
                                                     head=head,
                                                     aleatoric_head=aleatoric_head)
            else:
                (td_error,
                 weighted_error,
                 weighted_bayesian_loss,
                 weighted_error_summary,
                 bayesian_loss_summary,
                 # aleatoric_uncertainty, aleatoric_summary, aleatoric_normalised_summary,
                 # epistemic_uncertainty, epistemic_uncertainty2, epistemic_summary, epistemic_summary2,
                 # mean_hist, std_hist,
                 # regret_hist, inf_gain_hist, ids_hist,
                 #grad_var_summary,
                 #log_like_var_summary,
                 total_loss,
                 total_loss_summary, _) = bdqn.train(states=state_t,
                                                     rewards=reward_t,
                                                     next_states=state_tp1,
                                                     done=np.expand_dims(done_mask, axis=1),
                                                     weight=np.expand_dims(weight, axis=1),
                                                     kl_weight=kl_weight,
                                                     ep=episode_tensor,
                                                     head=head,
                                                     aleatoric_head=aleatoric_head)
            # summary_writer.add_summary(predictive_mean_summary, global_step)
            summary_writer.add_summary(weighted_error_summary, global_step)
            summary_writer.add_summary(bayesian_loss_summary, global_step)
            summary_writer.add_summary(total_loss_summary, global_step)
            # summary_writer.add_summary(aleatoric_summary, global_step)
            # summary_writer.add_summary(aleatoric_normalised_summary, global_step)
            # summary_writer.add_summary(epistemic_summary, global_step)
            #summary_writer.add_summary(grad_var_summary, global_step)
            #summary_writer.add_summary(log_like_var_summary, global_step)
            logging.info('Current Log Likelihood Loss (TD Error): %.4f', np.mean(np.abs(weighted_error)))
            logging.info('Current KL Weight: {:.7f}'.format(np.mean(np.abs(kl_weight))))
            logging.info('Current Bayesian loss: {:.4f}'.format(np.mean(np.abs(weighted_bayesian_loss))))
            # logging.info('Current Epistemic Uncertainty: {:.10f}'.format(np.mean(np.abs(epistemic_uncertainty))))
            # logging.info('Current Aleatoric Uncertainty: {:.7f}'.format(np.mean(np.abs(np.exp(aleatoric_uncertainty)))))
            logging.info('Current Total loss: {:.4f}'.format(np.mean(np.abs(total_loss))))
            if hparams.prioritized:
                memory.update_priorities(indices, np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())
        global_step += 1
    return global_step


def _step_nheads(environment, bdqn, memory, episode, global_step, hparams, exploration, head, aleatoric_head,
                 sample_number, summary_writer):
    """Runs a single step within an episode.

      Args:
        environment: molecules.Molecule; the environment to run on.
        bdqn: DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        hparams: HParams.
        exploration: Schedule used for exploration in the environment.
        head: Integer index of the DeepQNetwork head to use.

      Returns:
        molecules.Result object containing the result of the step.
      """
    # Compute the encoding for each valid action from the current state.
    # Calculate remaining steps (e.g. max 40 for constrained optimisation)
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    # Get the valid actions from the environment
    valid_actions = list(environment.get_valid_actions())
    # Get observations (np.array versions (from get fingerprint function)) of all possible new states (valid_actions) from the current state
    # These are the available actions from the current state
    observations = np.vstack(
        [np.append(b_deep_q_networks_ids_refactored.get_fingerprint(act, hparams), steps_left) for act in valid_actions])

    # Find the mean and variance of the Q-values for each valid action:
    if hparams.exp == 'ids' and not hparams.reward_eng:
        if episode > min(100, hparams.num_episodes / 10):
            action_index = bdqn.get_ids_action(observations, head, aleatoric_head,
                                               update_epsilon=exploration.value(episode))
            action = valid_actions[action_index]
            action_t_fingerprint = np.append(b_deep_q_networks_ids_refactored.get_fingerprint(action, hparams), steps_left)
            result = environment.step(action)
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
            action_tp1_fingerprints = np.vstack(
                [np.append(b_deep_q_networks_ids_refactored.get_fingerprint(act, hparams), steps_left) for act in
                 environment.get_valid_actions()])
    
            memory.add(
                obs_t=action_t_fingerprint,
                action=0,
                reward=result.reward,
                obs_tp1=action_tp1_fingerprints,
                done=float(result.terminated))
        else:
            # Chooses an action out of the above observations using epsilon greedy exploration
            action_index = bdqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))
            action = valid_actions[action_index]
            action_t_fingerprint = np.append(b_deep_q_networks_ids_refactored.get_fingerprint(action, hparams), steps_left)
            result = environment.step(action)
            # Recalculate the remaining number of steps
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
            # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
            action_tp1_fingerprints = np.vstack(
                [np.append(b_deep_q_networks_ids_refactored.get_fingerprint(act, hparams), steps_left) for act in
                  environment.get_valid_actions()])
            memory.add(
                obs_t=action_t_fingerprint,
                action=0,
                reward=result.reward,  # For ids action selection
                obs_tp1=action_tp1_fingerprints,
                done=float(result.terminated))
    elif hparams.exp == 'thompson' and not hparams.reward_eng:
        # Chooses an action out of the above observations using epsilon greedy exploration
        action_index = bdqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))
        action = valid_actions[action_index]
        action_t_fingerprint = np.append(b_deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
        result = environment.step(action)
        # Recalculate the remaining number of steps
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
        action_tp1_fingerprints = np.vstack(
            [np.append(b_deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])
        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward,  # For ids action selection
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

    if hparams.reward_eng:
        # Chooses an action out of the above observations using epsilon greedy exploration
        action_index = bdqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))
        action = valid_actions[action_index]
        # Get information terms
        (ids_scores, info_gains, e_uncert) = bdqn.get_ids_score(observations, head, aleatoric_head)
        ids_score = ids_scores[action_index]
        if info_gains is not None:
            info_gain = info_gains[action_index]
        if e_uncert is not None:
            e_uncert = e_uncert[action_index]
        #if e1_norm is not None:
        #    e1_norm = e1_norm[action_index]
        #if e2 is not None:
        #    e2 = e2
        #if e2_norm is not None:
        #    e2_norm = e2_norm
        if hparams.reward_eng == 'ids':
            print('ids: ' + str(ids_score) + '\t' + 'ids scaled: ' + str(ids_score * 1e-5)) # 1e-3, 1e-5
        else:
            ids_score = None
        if hparams.reward_eng == 'ids_2':
            print('ids_score_2: ' + str(ids_score * 1e-8))
        else:
            ids_score_2 = None
        if hparams.reward_eng == 'info_gain':
            print('info_gain: ' + str(info_gain))
        else:
            info_gain = None
        if hparams.reward_eng == 'e_uncert' or hparams.reward_eng == 'e_uncert2':
            print(hparams.reward_eng + ': ' + str(e_uncert))
        else:
            e_uncert = None

        action_t_fingerprint = np.append(b_deep_q_networks_ids_refactored.get_fingerprint(action, hparams), steps_left)
        # Take the step in the Molecule MDP environment and stores the Result in result
        result = environment.step(action, ids_score=ids_score, ids_score_norm=None, ids_score_2=None,
                                  ids_score_norm2=None,
                                  ids_true_norm=None, ids_true_e1_norm=None,
                                  info_gain_true_norm=None,
                                  info_gain_true_e1_norm=None,
                                  info_gain_norm=None, info_gain=info_gain, epistemic=e_uncert, e1_norm=None, e2=e_uncert,
                                  e2_norm=None,
                                  multi_obj=hparams.multi_obj, lr=hparams.local_reparam)
        # Recalculate the remaining number of steps
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
        action_tp1_fingerprints = np.vstack(
            [np.append(b_deep_q_networks_ids_refactored.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])
        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward_adjusted,  # For reward_eng
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

    # Returns the actual step taken in the environment
    return result


def _step(environment, bdqn, memory, episode, hparams, exploration, head):
    """Runs a single step within an episode.

      Args:
        environment: molecules.Molecule; the environment to run on.
        bdqn: DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        hparams: HParams.
        exploration: Schedule used for exploration in the environment.
        head: Integer index of the DeepQNetwork head to use.

      Returns:
        molecules.Result object containing the result of the step.
      """
    # Compute the encoding for each valid action from the current state.
    # Calculate remaining steps (e.g. max 40 for constrained optimisation)
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    # Get the valid actions from the environment
    valid_actions = list(environment.get_valid_actions())
    # Get observations (np.array versions (from get fingerprint function)) of all possible new states (valid_actions)
    observations = np.vstack(
        [np.append(b_deep_q_networks_uq.get_fingerprint(act, hparams), steps_left) for act in valid_actions])
    # Chooses an action out of the above observations using epsilon greedy exploration
    action = valid_actions[bdqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))]
    # Gets the fingerprint of the chosen action (new state) and appends the no. steps left to the array
    action_t_fingerprint = np.append(b_deep_q_networks_uq.get_fingerprint(action, hparams), steps_left)
    # Take the step in the Molecule MDP environment and stores the Result in result
    result = environment.step(action)
    # Recalculate the remaining number of steps
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    # Gets the fingerprints of all valid next actions (states)
    # from the new current state just calculated and adds the transitions to memory
    action_fingerprints = np.vstack(
        [np.append(b_deep_q_networks_uq.get_fingerprint(act, hparams), steps_left) for act in
         environment.get_valid_actions()])
    # we store the fingerprint of the action in obs_t so action
    # does not matter here.
    memory.add(
        obs_t=action_t_fingerprint,
        action=0,
        reward=result.reward,
        obs_tp1=action_fingerprints,
        done=float(result.terminated))
    # Returns the actual step taken
    return result


def run_bdqn_ids(multi_objective=False):
    """Run the training of Deep Q Network algorithm.

      Args:
        multi_objective: Boolean. Whether to run the multiobjective DQN.
      """
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, 'r') as f:
            hparams = b_deep_q_networks_uq.get_hparams(**json.load(f))
    else:
        hparams = b_deep_q_networks_uq.get_hparams()
    logging.info('HParams:\n%s',
                 '\n'.join(['\t%s: %s' % (key, value) for key, value in sorted(hparams.values().items())]))

    # TODO(zzp): merge single objective DQN to multi objective DQN.
    if multi_objective:
        environment = MultiObjectiveRewardMolecule(
            target_molecule=FLAGS.target_molecule,
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=False,
            allowed_ring_sizes={3, 4, 5, 6},
            max_steps=hparams.max_steps_per_episode)

        bdqn = b_deep_q_networks_uq.MultiObjectiveDeepBBQNetwork(
            objective_weight=np.array([[FLAGS.similarity_weight], [1 - FLAGS.similarity_weight]]),
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(b_deep_q_networks_uq.multi_layer_bayesian_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0)
    else:
        environment = TargetWeightMolecule(
            target_weight=FLAGS.target_weight,
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=hparams.allow_bonds_between_rings,
            allowed_ring_sizes=set(hparams.allowed_ring_sizes),
            max_steps=hparams.max_steps_per_episode)

        bdqn = b_deep_q_networks_ids.DeepBBQNetwork(
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(b_deep_q_networks_ids.multi_layer_bayesian_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0)

    run_training(
        hparams=hparams,
        environment=environment,
        bdqn=bdqn)

    core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


def main(argv):
    del argv  # unused.
    run_bdqn(FLAGS.multi_objective)


if __name__ == '__main__':
    app.run(main)
