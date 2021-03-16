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
import b_deep_q_networks_ps
import molecules as molecules_mdp
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
    tf.reset_default_graph()
    with tf.Session() as sess:

        # Build the dqn
        bdqn.build()

        # Define a model saver
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)

        # The schedule for the epsilon in epsilon greedy policy.
        exploration = schedules.PiecewiseSchedule(
            [(0, 1.0), (int(hparams.num_episodes / 2), 0.1), (hparams.num_episodes, 0.01)], outside_value=0.01)

        # Define the memory variable as Prioritised Experience Replay Buffer from baselines modules
        if hparams.prioritized:
            memory = replay_buffer.PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
            beta_schedule = schedules.LinearSchedule(hparams.num_episodes, initial_p=hparams.prioritized_beta,
                                                     final_p=0)
        else:
            memory = replay_buffer.ReplayBuffer(hparams.replay_buffer_size)
            beta_schedule = None

        # The schedule for the Bayesian Loss
        phi_kl_schedule = schedules.LinearSchedule(hparams.num_episodes, 1e-5, initial_p=1e-6)
        theta_kl_schedule = schedules.LinearSchedule(hparams.num_episodes, 1e-4, initial_p=1e-4)

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
                exploration=exploration,
                beta_schedule=beta_schedule,
                # kl_weight_schedule=KLSchedule(hparams, ep=episode, weighting_type=hparams.weighting_type)
                phi_kl_schedule=phi_kl_schedule,
                theta_kl_schedule=theta_kl_schedule)
            print(bdqn.built_flag)
            print(bdqn.built_number)
            # Update the target network every 'hparams.update_frequency' episodes
            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(bdqn.update_op)
                sess.run(bdqn.update_op_theta)
            # Save the model every 'hparams.save_frequency' episodes (currently set to every 10 updates)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(sess, os.path.join(FLAGS.model_dir, 'ckpt'), global_step=global_step)


def _episode(environment, bdqn, memory, episode, global_step, hparams, summary_writer, exploration, beta_schedule,
             phi_kl_schedule, theta_kl_schedule):
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

      Returns:
        Updated global_step.
      """
    episode_start_time = time.time()
    environment.initialize()
    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
    else:
        head = 0
    for step in range(hparams.max_steps_per_episode):
        # Takes a step
        result = _step(
            environment=environment,
            bdqn=bdqn,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head)
        if step == hparams.max_steps_per_episode - 1:
            episode_summary = bdqn.log_result(result.state, result.reward)
            summary_writer.add_summary(episode_summary, global_step)
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes, time.time() - episode_start_time)
            logging.info('SMILES: %s', result.state)
            # Use %s since reward can be a tuple or a float number.
            logging.info('The reward is: %s\n', str(result.reward))
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
            phi_kl_weight = phi_kl_schedule.value(episode)
            theta_kl_weight = theta_kl_schedule.value(episode)
            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
            episode_tensor = np.array([episode]).astype(float)
            (td_error,
             td_error_theta,
             weighted_error,
             weighted_error_theta,
             weighted_phi_kl_loss,
             weighted_theta_kl_loss,
             weighted_error_summary,
             weighted_phi_kl_loss_summary,
             weighted_theta_kl_loss_summary,
             total_loss,
             total_loss_summary, ps_difference,
             u_plex, s_plex,
             grad_var1, grad_var2,
             llvar1, llvar2, _) = bdqn.train(states=state_t,
                                             rewards=reward_t,
                                             next_states=state_tp1,
                                             done=np.expand_dims(done_mask, axis=1),
                                             weight=np.expand_dims(weight, axis=1),
                                             phi_kl_weight=phi_kl_weight,
                                             theta_kl_weight=theta_kl_weight,
                                             ep=episode_tensor)
            summary_writer.add_summary(weighted_error_summary, global_step)
            summary_writer.add_summary(weighted_phi_kl_loss_summary, global_step)
            summary_writer.add_summary(weighted_theta_kl_loss_summary, global_step)
            summary_writer.add_summary(total_loss_summary, global_step)
            summary_writer.add_summary(ps_difference, global_step)
            summary_writer.add_summary(u_plex, global_step)
            summary_writer.add_summary(s_plex, global_step)
            summary_writer.add_summary(grad_var1, global_step)
            summary_writer.add_summary(llvar1, global_step)
            summary_writer.add_summary(grad_var2, global_step)
            summary_writer.add_summary(llvar2, global_step)
            # logging.info('Current shape: {}'.format(out_shape))
            logging.info('Current Log Likelihood Loss (TD Error): %.4f', np.mean(np.abs(weighted_error)))
            logging.info('Current Theta-Log Likelihood Loss (TD Error): %.4f', np.mean(np.abs(weighted_error_theta)))
            logging.info('Current Phi KL Weight: {:.7f}'.format(np.mean(np.abs(phi_kl_weight))))
            logging.info('Current Theta KL Weight: {:.7f}'.format(np.mean(np.abs(theta_kl_weight))))
            logging.info('Current Phi KL loss: {:.4f}'.format(np.mean(np.abs(weighted_phi_kl_loss))))
            logging.info('Current Theta KL loss: {:.4f}'.format(np.mean(np.abs(weighted_theta_kl_loss))))
            logging.info('Current Total loss: {:.4f}'.format(np.mean(np.abs(total_loss))))
            if hparams.prioritized:
                memory.update_priorities(indices, np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())
        global_step += 1
    return global_step



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
        [np.append(b_deep_q_networks_ps.get_fingerprint(act, hparams), steps_left) for act in valid_actions])
    # Chooses an action out of the above observations using epsilon greedy exploration
    action = bdqn.get_action(observations, head=head, episode=episode, update_epsilon=exploration.value(episode))
    action = valid_actions[action]
    # Gets the fingerprint of the chosen action (new state) and appends the no. steps left to the array
    action_t_fingerprint = np.append(b_deep_q_networks_ps.get_fingerprint(action, hparams), steps_left)
    # Take the step in the Molecule MDP environment and stores the Result in result
    result = environment.step(action)
    # Recalculate the remaining number of steps
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    # Gets the fingerprints of all valid next actions (states)
    # from the new current state just calculated and adds the transitions to memory
    action_fingerprints = np.vstack(
        [np.append(b_deep_q_networks_ps.get_fingerprint(act, hparams), steps_left) for act in
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


def run_bdqn(multi_objective=False):
    """Run the training of Deep Q Network algorithm.

      Args:
        multi_objective: Boolean. Whether to run the multiobjective DQN.
      """
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, 'r') as f:
            hparams = b_deep_q_networks2.get_hparams(**json.load(f))
    else:
        hparams = b_deep_q_networks2.get_hparams()
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

        bdqn = b_deep_q_networks2.MultiObjectiveDeepBBQNetwork(
            objective_weight=np.array([[FLAGS.similarity_weight], [1 - FLAGS.similarity_weight]]),
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(b_deep_q_networks2.multi_layer_bayesian_model, hparams=hparams),
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

        bdqn = b_deep_q_networks2.DeepBBQNetwork(
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(b_deep_q_networks_ps.multi_layer_bayesian_model, hparams=hparams),
            # q_fn=MLP(hparams),
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
