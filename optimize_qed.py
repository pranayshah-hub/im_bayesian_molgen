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

"""Optimize the QED of a molecule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags

from rdkit import Chem
from rdkit.Chem import QED
from tensorflow import gfile
from bbb_utils import *

# import deep_q_networks
#import deep_q_networks_ids
# import b_deep_q_networks
# import b_deep_q_networks_correct
#import b_deep_q_networks_ps
import b_deep_q_networks_ids_refactored
import molecules_ids as molecules_mdp_ids
import molecules as molecules_mdp

# import run_dqn
#import run_dqn_ids
#import run_bdqn
#import run_bdqn_ps
import run_bdqn_ids_refactored
import core

#flags.DEFINE_float('gamma', 0.999, 'discount')
flags.DEFINE_string('model_type', 'bdqn_ids', 'Enter the model type.')
flags.DEFINE_integer('number_of_trials', '5', 'Number of trials.')
FLAGS = flags.FLAGS

import tensorflow as tf
import numpy as np

class Molecule(molecules_mdp.Molecule):
    """QED reward Molecule."""

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        try:
            qed = QED.qed(molecule)
        except ValueError:
            qed = 0
        # return qed * 0.999 ** (self.max_steps - self._counter)
        return qed


class Molecule_IDS(molecules_mdp_ids.Molecule):
    """QED reward Molecule."""

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        try:
            qed = QED.qed(molecule)
        except ValueError:
            qed = 0
        # return qed * 0.999 ** (self.max_steps - self._counter)
        return qed


def main(argv):
    del argv

    if FLAGS.model_type == 'dqn':
        if FLAGS.hparams is not None:
            with gfile.Open(FLAGS.hparams, 'r') as f:
                hparams = deep_q_networks.get_hparams(**json.load(f))
        else:
            hparams = deep_q_networks.get_hparams()

        for t in range(FLAGS.number_of_trials):
            tf.reset_default_graph()

            environment = Molecule(
                atom_types=set(hparams.atom_types),
                init_mol=FLAGS.start_molecule,
                allow_removal=hparams.allow_removal,
                allow_no_modification=hparams.allow_no_modification,
                allow_bonds_between_rings=hparams.allow_bonds_between_rings,
                allowed_ring_sizes=set(hparams.allowed_ring_sizes),
                max_steps=hparams.max_steps_per_episode)

            dqn = deep_q_networks.DeepQNetwork(
                input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
                # hparams=hparams,
                q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
                optimizer=hparams.optimizer,
                grad_clipping=hparams.grad_clipping,
                num_bootstrap_heads=hparams.num_bootstrap_heads,
                gamma=hparams.gamma,
                epsilon=1.0)

            run_dqn.run_training(
                hparams=hparams,
                environment=environment,
                dqn=dqn)

        core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'bsdqn_config.json'))

    if FLAGS.model_type == 'dqn_ids':
        if FLAGS.hparams is not None:
            with gfile.Open(FLAGS.hparams, 'r') as f:
                hparams = deep_q_networks_ids.get_hparams(**json.load(f))
        else:
            hparams = deep_q_networks_ids.get_hparams()

        for t in range(FLAGS.number_of_trials):
            tf.reset_default_graph()

            environment = Molecule_IDS(
                atom_types=set(hparams.atom_types),
                init_mol=FLAGS.start_molecule,
                allow_removal=hparams.allow_removal,
                allow_no_modification=hparams.allow_no_modification,
                allow_bonds_between_rings=hparams.allow_bonds_between_rings,
                allowed_ring_sizes=set(hparams.allowed_ring_sizes),
                max_steps=hparams.max_steps_per_episode)

            dqn_ids = deep_q_networks_ids.DeepQNetworkIDS(
                ipt_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
                hparams=hparams,
                q_fn=functools.partial(deep_q_networks_ids.multi_layer_model_ids, hparams=hparams),
                optimizer=hparams.optimizer,
                grad_clipping=hparams.grad_clipping,
                num_bootstrap_heads=hparams.num_bootstrap_heads,
                gamma=hparams.gamma,
                epsilon=1.0,
                exploration='ids')

            run_dqn_ids.run_training(
            hparams=hparams,
            environment=environment,
            dqn_ids=dqn_ids)

        core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))

    elif FLAGS.model_type == 'bdqn':
        if FLAGS.hparams is not None:
            with gfile.Open(FLAGS.hparams, 'r') as f:
                hparams = b_deep_q_networks_correct.get_hparams(**json.load(f))
        else:
            hparams = b_deep_q_networks_correct.get_hparams()

        for t in range(FLAGS.number_of_trials):
            tf.reset_default_graph()

            environment = Molecule(atom_types=set(hparams.atom_types),
                                   init_mol=FLAGS.start_molecule,
                                   allow_removal=hparams.allow_removal,
                                   allow_no_modification=hparams.allow_no_modification,
                                   allow_bonds_between_rings=hparams.allow_bonds_between_rings,
                                   allowed_ring_sizes=set(hparams.allowed_ring_sizes),
                                   max_steps=hparams.max_steps_per_episode)

            bdqn = b_deep_q_networks_correct.DeepBBQNetwork(hparams=hparams,
                                                            ipt_shape=(
                                                            hparams.batch_size, hparams.fingerprint_length + 1),
                                                            # local_reparameterisation=hparams.local_reparam,
                                                            optimizer=hparams.optimizer,
                                                            grad_clipping=hparams.grad_clipping,
                                                            num_bootstrap_heads=hparams.num_bootstrap_heads,
                                                            gamma=hparams.gamma,
                                                            epsilon=1.0)

            run_bdqn.run_training(
                hparams=hparams,
                environment=environment,
                bdqn=bdqn)

        core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))

    elif FLAGS.model_type == 'bdqn_ps':
        if FLAGS.hparams is not None:
            with gfile.Open(FLAGS.hparams, 'r') as f:
                hparams = b_deep_q_networks_ps.get_hparams(**json.load(f))
        else:
            hparams = b_deep_q_networks_ps.get_hparams()

        for t in range(FLAGS.number_of_trials):
            tf.reset_default_graph()

            environment = Molecule(atom_types=set(hparams.atom_types),
                                   init_mol=FLAGS.start_molecule,
                                   allow_removal=hparams.allow_removal,
                                   allow_no_modification=hparams.allow_no_modification,
                                   allow_bonds_between_rings=hparams.allow_bonds_between_rings,
                                   allowed_ring_sizes=set(hparams.allowed_ring_sizes),
                                   max_steps=hparams.max_steps_per_episode)

            bdqn = b_deep_q_networks_ps.DeepBBQNetwork(hparams=hparams,
                                                       ipt_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
                                                       local_reparameterisation=hparams.local_reparam,
                                                       optimizer=hparams.optimizer,
                                                       grad_clipping=hparams.grad_clipping,
                                                       num_bootstrap_heads=hparams.num_bootstrap_heads,
                                                       gamma=hparams.gamma,
                                                       epsilon=1.0)

            run_bdqn_ps.run_training(
            hparams=hparams,
            environment=environment,
            bdqn=bdqn)

        core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


    elif FLAGS.model_type == 'bdqn_ids':
        if FLAGS.hparams is not None:
            with gfile.Open(FLAGS.hparams, 'r') as f:
                hparams = b_deep_q_networks_ids_refactored.get_hparams(**json.load(f))
        else:
            hparams = b_deep_q_networks_ids_refactored.get_hparams()

        for t in range(FLAGS.number_of_trials):
            tf.reset_default_graph()

            environment = Molecule(atom_types=set(hparams.atom_types),  # Molecules_IDS for reward_eng
                                   init_mol=FLAGS.start_molecule,
                                   allow_removal=hparams.allow_removal,
                                   allow_no_modification=hparams.allow_no_modification,
                                   allow_bonds_between_rings=hparams.allow_bonds_between_rings,
                                   allowed_ring_sizes=set(hparams.allowed_ring_sizes),
                                   max_steps=hparams.max_steps_per_episode)
            # num_obj=2)

            bdqn = b_deep_q_networks_ids_refactored.DeepBBQNetwork(hparams=hparams,
                                                        ipt_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
                                                        # local_reparameterisation=hparams.local_reparam,
                                                        optimizer=hparams.optimizer,
                                                        grad_clipping=hparams.grad_clipping,
                                                        num_bootstrap_heads=hparams.num_bootstrap_heads,
                                                        gamma=hparams.gamma,
                                                        epsilon=1.0)

            run_bdqn_ids_refactored.run_training(
            hparams=hparams,
            environment=environment,
            bdqn=bdqn)

        core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


if __name__ == '__main__':
    app.run(main)

