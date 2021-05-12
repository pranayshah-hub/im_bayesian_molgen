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

import tensorflow as tf


def compute_log_predictive_variance(inputs_list, aleatory_head):

    # Calculate log_pred_var per sample.
    log_pred_var_all = [tf.gather(inputs_list[i], aleatory_head, axis=-1) for i in range(len(inputs_list))]

    # Calculate mean log_pred_var across all samples. Left with shape [num_actions, 1]
    log_pred_var = tf.reduce_mean(tf.stack(log_pred_var_all, axis=1), axis=1)

    return log_pred_var

def compute_aleatory(log_pred_var):

    eps = 1e-9

    # Return distribution variance (aleatoric variance)
    a_uncert = tf.exp(log_pred_var)
    sum_a_uncert = tf.reduce_sum(a_uncert)
    no_actions = a_uncert.shape[0].value

    # Compute the normalised aleatoric variance per action for the IDS ratio
    if no_actions is not None:
        a_uncert_norm = a_uncert / (eps + (1 / no_actions) * sum_a_uncert)
    else:
        a_uncert_norm = a_uncert

    return a_uncert_norm

def compute_epistemic(inputs_mean, inputs_list, head, layers_dict, dense_layers, n_samples, e_type):

    q_values_all = [tf.gather(inputs_list[i], head, axis=-1) for i in range(len(inputs_list))]
    q_values_all = tf.stack(q_values_all, axis=1)

    # mu = tf.reduce_mean(q_values_all, axis=1)
    mu = inputs_mean

    if e_type == 1:
        # Epistemic uncertainty per action
        mu_centralised = q_values_all - tf.expand_dims(mu, axis=-1)  # axis = -1
        e_uncert = tf.reduce_mean(tf.square(mu_centralised), axis=1)
        std = tf.sqrt(e_uncert)

    elif e_type == 2:
        eps = 1e-9

        w_list = list()  # [5, num_samples]
        b_list = list()  # [5, num_samples]

        for i in range(len(dense_layers)):
            w_list.append(layers_dict['dense_{}'.format(i)].w_uncertainty)
            b_list.append(layers_dict['dense_{}'.format(i)].b_uncertainty)

        w_list.append(layers_dict['dense_final'].w_uncertainty)
        b_list.append(layers_dict['dense_final'].b_uncertainty)

        squared_w_uncert = [[tf.square(w_list[i][j]) for j in range(n_samples)] for i in range(len(w_list))]
        squared_b_uncert = [[tf.square(b_list[i][j]) for j in range(n_samples)] for i in range(len(b_list))]

        normalising_constant = sum([n_samples * layers_dict[i].w_dims[0] * layers_dict[i].w_dims[1] for i in range(
            len(layers_dict))]) + sum([n_samples * layers_dict[i].b_dims[-1] for i in range(len(layers_dict))])

        sum_uncert = sum([tf.reduce_sum(tf.stack(squared_w_uncert[i], axis=-1)) for i in
                          range(len(squared_w_uncert))])
        sum_uncert += sum([tf.reduce_sum(tf.stack(squared_b_uncert[i], axis=-1)) for i in
                           range(len(squared_b_uncert))])

        e_uncert = sum_uncert / (normalising_constant + eps)
        std = tf.sqrt(e_uncert)

    return mu, e_uncert, std

def compute_epistemic_clt():
    return mu, e_uncert, std

def compute_epistemic_bs(inputs_list):

    mu = None

    # Epistemic uncertainty per action
    e_uncert = None

    std = None

    return mu, e_uncert, std

def compute_regret(mu, std, num_sigma):

    regret = tf.reduce_max(mu + num_sigma * std, axis=-1, keepdims=True) - (mu - num_sigma * std)

    return regret

def compute_ig(e_uncert, a_uncert_norm):

    eps = 1e-9

    ig = tf.log(1 + e_uncert / a_uncert_norm) + eps

    return ig

def compute_ids(regret, ig):

    ids = tf.square(regret) / ig

    return ids

