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
import deep_q_networks_ids
import molecules_ids as molecules_mdp
import molecules_py
import core

flags.DEFINE_string('model_dir',
                    '',
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
        lower, upper = self.target_weight - 25, self.target_weight + 25
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

    def __init__(self, hparams, uniform=True, outside_value=None):
        self._max_steps_per_episode = hparams.max_steps_per_episode
        self._update_frequence = hparams.update_frequence
        self._num_episodes = hparams.num_episodes
        self.M = float(self._max_steps_per_episode / self._update_frequence * self._num_episodes)
        self._outside_value = outside_value
        self._uniform = uniform

    def value(self, step):
        if self._uniform:
            return 1.0 / self.M
        else:
            return (2 ** (self.M - step)) / (2 ** self.M - 1)



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
def run_training(hparams, environment, dqn_ids):
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
    tf.reset_default_graph()
    with tf.Session() as sess:

        # Build the dqn
        dqn_ids.build()

        # Define a model saver
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)

        # The schedule for the epsilon in epsilon greedy policy.
        if hparams.exp == 'thompson'
            exploration = schedules.PiecewiseSchedule(
                [(0, 1.0), (int(hparams.num_episodes / 2), 0.1), (hparams.num_episodes, 0.01)], outside_value=0.01)
        else:
            exploration = schedules.PiecewiseSchedule(
                [(0, 1.0), (int(hparams.num_episodes * 0.4), 0.1), (hparams.num_episodes, 0.01)], outside_value=0.01)

        # Define the memory variable as Prioritised Experience Replay Buffer from baselines modules
        if hparams.prioritized:
            memory = replay_buffer.PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
            beta_schedule = schedules.LinearSchedule(hparams.num_episodes, initial_p=hparams.prioritized_beta,
                                                     final_p=0)
        else:
            memory = replay_buffer.ReplayBuffer(hparams.replay_buffer_size)
            beta_schedule = None

        # Run the session with the global variables initialiser
        sess.run(tf.global_variables_initializer())

        # Update the dqn for the first time after initialising all variables
        sess.run(dqn_ids.update_op)

        # Run an episode
        global_step = 0
        for episode in range(hparams.num_episodes):
            global_step = _episode(
                environment=environment,
                dqn_ids=dqn_ids,
                memory=memory,
                episode=episode,
                global_step=global_step,
                hparams=hparams,
                summary_writer=summary_writer,
                exploration=exploration,
                beta_schedule=beta_schedule)
            # Update the target network every 'hparams.update_frequency' episodes
            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(dqn_ids.update_op)
            # Save the model every 'hparams.save_frequency' episodes (currently set to every 10 updates)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(sess, os.path.join(FLAGS.model_dir, 'ckpt'), global_step=global_step)


def _episode(environment, dqn_ids, memory, episode, global_step, hparams, summary_writer, exploration, beta_schedule):
    """Runs a single episode.

      Args:
        environment: molecules.Molecule; the environment to run on.
        dqn_ids: IDS-based Bootstrapped-DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        global_step: Integer global step; the total number of steps across all
          episodes.
        hparams: HParams.
        summary_writer: FileWriter used for writing Summary protos.
        exploration: Schedule used for exploration in the environment.
        beta_schedule: Schedule used for prioritized replay buffers.

      Returns:
        Updated global_step.
      """
    episode_start_time = time.time()
    environment.initialize()
    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
        aleatoric_head = 12
    else:
        head = 0
        aleatoric_head = 1
    for step in range(hparams.max_steps_per_episode):
        # Takes a step
        result = _step(
            environment=environment,
            dqn_ids=dqn_ids,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head,
            aleatoric_head=aleatoric_head)
        if step == hparams.max_steps_per_episode - 1:
            if hparams.re:
                episode_summary, adjusted_episode_summary = dqn_ids.log_result(result.state, result.reward,
                                                                               result.reward_adjusted)
                summary_writer.add_summary(adjusted_episode_summary, global_step)
            else:
                episode_summary = dqn_ids.log_result(result.state, result.reward, None)
            summary_writer.add_summary(episode_summary, global_step)
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes, time.time() - episode_start_time)
            logging.info('SMILES: %s\n', result.state)
            # Use %s since reward can be a tuple or a float number.
            logging.info('The reward is: %s', str(result.reward))
        # Waits for the replay buffer to contain at least 50 * max_steps_per_episode t
        # transitions before sampling and updating the dqn
        if (episode > min(50, hparams.num_episodes / 10)) and (global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                (state_t, _, reward_t, state_tp1, done_mask, weight, indices) = memory.sample(hparams.batch_size,
                                                                                              beta=beta_schedule.value(
                                                                                                  episode))
            else:
                (state_t, std, reward_t, state_tp1, done_mask) = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
            (td_error, predictive_mean_summary, error_summary,
             aleatoric_uncertainty, aleatoric_summary,
             epistemic_uncertainty, epistemic_summary,
             q_values, q_values_all, _) = dqn_ids.train(head=head,
                                                        aleatoric_head=aleatoric_head,
                                                        states=state_t,
                                                        rewards=reward_t,
                                                        next_states=state_tp1,
                                                        done=np.expand_dims(done_mask, axis=1),
                                                        weight=np.expand_dims(weight, axis=1))
            summary_writer.add_summary(predictive_mean_summary, global_step)
            summary_writer.add_summary(error_summary, global_step)
            summary_writer.add_summary(aleatoric_summary, global_step)
            summary_writer.add_summary(epistemic_summary, global_step)
            logging.info('Current TD error: %.4f', np.mean(np.abs(td_error)))
            logging.info('Current Epistemic Uncertainty: {:.7f}'.format(np.mean(np.abs(epistemic_uncertainty))))
            logging.info('Current Aleatoric Uncertainty: {:.7f}'.format(np.mean(np.abs(aleatoric_uncertainty))))
            if hparams.prioritized:
                memory.update_priorities(indices, np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())
        global_step += 1
    return global_step


def _step_nheads(environment, dqn_ids, memory, episode, hparams, exploration, head, aleatoric_head):
    """Runs a single step within an episode.

      Args:
        environment: molecules.Molecule; the environment to run on.
        dqn_ids: IDS-based Bootstrapped-DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        hparams: HParams.
        exploration: Schedule used for exploration in the environment.
        head: Integer index of the DeepQNetwork head to use.
        aleatoric_head: Integer index of the DeepQNetwork head that predicts aleatoric uncertainty

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
        [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in valid_actions])

    # Find the mean and variance of the Q-values for each valid action:
    if episode > min(500, hparams.num_episodes / 10):
        a = dqn_ids.get_ids_action(observations, aleatoric_head, update_epsilon=exploration.value(episode))
        action = valid_actions[a]
        action_t_fingerprint = np.append(deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
        result = environment.step(action)
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        action_tp1_fingerprints = np.vstack(
            [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])

        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward,
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

        return result

    else:
        # Chooses an action out of the above observations using epsilon greedy exploration
        action = valid_actions[dqn_ids.get_action(observations, head=head, update_epsilon=exploration.value(episode))]
        action_t_fingerprint = np.append(deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)

        # Take the step in the Molecule MDP environment and stores the Result in result
        result = environment.step(action)

        # Recalculate the remaining number of steps
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken

        # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
        action_tp1_fingerprints = np.vstack(
            [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])

        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward,
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

        # Returns the actual step taken in the environment
        return result


def _step(environment, dqn_ids, memory, episode, hparams, exploration, head, aleatoric_head):
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
        [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in valid_actions])

    # Find the mean and variance of the Q-values for each valid action:
    if hparams.exp == 'ids':
        if episode > min(100, hparams.num_episodes / 10): # If IDS
            action_index = bdqn.get_ids_action(observations, head, aleatoric_head, update_epsilon=exploration.value(episode))
            action = valid_actions[action_index]
            action_t_fingerprint = np.append(b_deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
            result = environment.step(action)
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
            action_tp1_fingerprints = np.vstack(
                [np.append(b_deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
                  environment.get_valid_actions()])

            memory.add(
                obs_t=action_t_fingerprint,
                action=0,
                reward=result.reward,
                obs_tp1=action_tp1_fingerprints,
                done=float(result.terminated))
        else: # If Thompson
            # Chooses an action out of the above observations using epsilon greedy exploration
            action_index = dqn_ids.get_action(observations, head=head, aleatoric_head=aleatoric_head,update_epsilon=exploration.value(episode))
            action = valid_actions[action_index]
            action_t_fingerprint = np.append(deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
            result = environment.step(action)
            # Recalculate the remaining number of steps
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
            # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
            action_tp1_fingerprints = np.vstack(
                [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
                  environment.get_valid_actions()])
            memory.add(
                obs_t=action_t_fingerprint,
                action=0,
                reward=result.reward,  # For ids action selection
                obs_tp1=action_tp1_fingerprints,
                done=float(result.terminated))
    else:  # If Thompson
        # Chooses an action out of the above observations using epsilon greedy exploration
        action_index = dqn_ids.get_action(observations, head=head, aleatoric_head=aleatoric_head,
                                          update_epsilon=exploration.value(episode))
        action = valid_actions[action_index]
        action_t_fingerprint = np.append(deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
        result = environment.step(action)
        # Recalculate the remaining number of steps
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
        action_tp1_fingerprints = np.vstack(
            [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])
        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward,  # For ids action selection
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

    if hparams.re:
        # Chooses an action out of the above observations using epsilon greedy exploration
        action_index = dqn_ids.get_action(observations, head=head, aleatoric_head=aleatoric_head,
                                          update_epsilon=exploration.value(episode))
        action = valid_actions[action_index]
        # Get information terms
        (ids_scores, info_gains, e1) = dqn_ids.get_ids_score(observations, head, aleatoric_head)
        ids_score = ids_scores[action_index]
        ids_score_norm = ids_score
        ids_score_2 = ids_score
        ids_score_norm2 = ids_score
        ids_true_norm = ids_score
        ids_true_e1_norm = ids_score
        if info_gains is not None:
            info_gain = info_gains[action_index]
            info_gain_norm = info_gain
            info_gain_true_norm = info_gain
            info_gain_true_e1_norm = info_gain
        else:
            info_gain = None
            info_gain_norm = None
            info_gain_true_norm = None
            info_gain_true_e1_norm = None
        if hparams.reward_eng == 'ids':
            print('ids: ' + str(ids_score) + '\t' + 'ids scaled: ' + str(ids_score * 1e-2))
        else:
            ids_score = None
        if hparams.reward_eng == 'ids_norm':
            print('ids_norm: ' + str(ids_score_norm))
        else:
            ids_score_norm = None
        if hparams.reward_eng == 'ids_2':
            print('ids_score_2: ' + str(ids_score_2 * 1e-8))
        else:
            ids_score_2 = None
        if hparams.reward_eng == 'ids_norm2':
            print('ids_score_norm2: ' + str(ids_score_norm2))
        else:
            ids_score_norm2 = None
        if hparams.reward_eng == 'ids_true_norm':
            print('ids_true_norm: ' + str(ids_true_norm))
        else:
            ids_true_norm = None
        if hparams.reward_eng == 'ids_true_e1_norm':
            print('ids_true_e1_norm: ' + str(ids_true_e1_norm))
        else:
            ids_true_e1_norm = None

        action_t_fingerprint = np.append(deep_q_networks_ids.get_fingerprint(action, hparams), steps_left)
        # Take the step in the Molecule MDP environment and stores the Result in result
        result = environment.step(action, ids_score=ids_score, ids_score_norm=ids_score_norm, ids_score_2=ids_score_2,
                                  ids_score_norm2=ids_score_norm2,
                                  ids_true_norm=ids_true_norm, ids_true_e1_norm=ids_true_e1_norm,
                                  multi_obj=hparams.multi_obj)
        # Recalculate the remaining number of steps
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        # Gets the fingerprints of all valid next actions (states) from the new current state just calculated and adds the transitions to memory
        action_tp1_fingerprints = np.vstack(
            [np.append(deep_q_networks_ids.get_fingerprint(act, hparams), steps_left) for act in
             environment.get_valid_actions()])
        memory.add(
            obs_t=action_t_fingerprint,
            action=0,
            reward=result.reward_adjusted,  # For reward_eng
            obs_tp1=action_tp1_fingerprints,
            done=float(result.terminated))

    # Returns the actual step taken
    return result



def run_dqn_ids(multi_objective=False):
    """Run the training of Deep Q Network algorithm.

      Args:
        multi_objective: Boolean. Whether to run the multiobjective DQN.
      """
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, 'r') as f:
            hparams = deep_q_networks_ids.get_hparams(**json.load(f))
    else:
        hparams = deep_q_networks_ids.get_hparams()
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

        dqn_ids = deep_q_networks_ids.MultiObjectiveDeepQNetworkIDS(
            objective_weight=np.array([[FLAGS.similarity_weight], [1 - FLAGS.similarity_weight]]),
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(deep_q_networks_ids.multi_layer_model, hparams=hparams),
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

        dqn_ids = deep_q_networks_ids.DeepQNetworkIDS(
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(deep_q_networks_ids.multi_layer_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0)

    run_training(
        hparams=hparams,
        environment=environment,
        dqn_ids=dqn_ids)

    core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


def main(argv):
    del argv  # unused.
    run_bdqn(FLAGS.multi_objective)


if __name__ == '__main__':
    app.run(main)
