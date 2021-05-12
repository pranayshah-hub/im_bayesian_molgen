import os
import statistics
import collections
import operator

# import tensorflow as tf

# For reading TF Events files.
# def summarise_good_molecules(file_path_list):
#     molecules = collections.defaultdict(list)
#     for file_path in file_path_list:
#         for summary in tf.train.summary_iterator(file_path):
#             # print(summary.summary.value)
#
#             found = False
#
#             for e in summary.summary.value:
#                 if e.tag == "bdqn_2/summaries/SMILES":
#                     smiles = repr(e.tensor.string_val)
#                     found = True
#                 if e.tag == "bdqn_2/summaries/reward":
#                     reward = float(e.simple_value)
#                     found = True
#             if found:
#                 # print(smiles, reward)
#                 molecules[smiles].append(reward)
#
#         for k, v in molecules.items():
#             if len(v) > 1:
#                 m = statistics.mean(v)
#                 s = statistics.stdev(v)
#                 mx = max(v)
#             else:
#                 m = v[0]
#                 s = 0.0
#                 mx = v[0]
#             # molecules[k] = (m, s, mx)
#             molecules[k] = m
#             # print(k, (m, s, mx))
#
#     molecules_sorted = sorted(molecules, key=molecules.get)
#
#     for m in molecules_sorted[-20:]:
#         print(m, molecules[m])
#     print()
#
#
# # summarise_good_molecules(
# #             ["/data/PycharmProjects/B2Q_code/bdqn/events.out.tfevents.1618586241.beilun.doc.ic.ac.uk",
# #              ])
# # summarise_good_molecules(
# #             ["/data/PycharmProjects/B2Q_code/bdqn2/events.out.tfevents.1618617464.beilun.doc.ic.ac.uk",
# #              ])


def summarise_molecule_output(folder,
                              top_k=50):
    file_to_molecules = dict()

    molecule_to_reward = dict()

    file_names = os.listdir(folder)
    file_names = [file_name for file_name in file_names if "molecule_output" in file_name]
    for file_name in file_names:
        file_path = folder + "/" + file_name

        molecule_to_reward[file_name] = dict()

        with open(file_path, "r") as fp:
            for row in fp:
                clean_row = row.strip().split("\t")

                molecule_to_reward[file_name][clean_row[0]] = float(clean_row[1])

        molecules_sorted = sorted(molecule_to_reward[file_name], key=molecule_to_reward[file_name].get)

        file_to_molecules[file_name] = molecules_sorted

    means = []
    stds = []
    maxes = []
    mins = []
    for k in range(top_k):
        m_id = - top_k + k
        reward_list = list()
        for file_name in file_names:
            m = file_to_molecules[file_name][m_id]
            reward_list.append(molecule_to_reward[file_name][m])
        if len(reward_list) == 1:
            mean_reward = reward_list[0]
            std_reward = 0.0
            max_reward = reward_list[0]
            min_reward = reward_list[0]
        elif len(reward_list) > 1:
            mean_reward = statistics.mean(reward_list)
            std_reward = statistics.stdev(reward_list)
            max_reward = max(reward_list)
            min_reward = min(reward_list)
        else:
            raise ValueError
        print(mean_reward, std_reward, max_reward, min_reward)
        means.append(mean_reward)
        stds.append(std_reward)
        maxes.append(max_reward)
        mins.append(min_reward)

    return means, stds, maxes, mins


# Point this to a folder, and it will read and average all the *molecule_output* files.
exp1 = summarise_molecule_output("/data/PycharmProjects/B2Q_code/dqn_56/",
                          top_k=5)
print()
exp2 = summarise_molecule_output("/data/PycharmProjects/B2Q_code/dqn_567/",
                          top_k=5)
print()
summarise_molecule_output("/data/PycharmProjects/B2Q_code/dqn_34567/",
                          top_k=5)

print()
summarise_molecule_output("/data/PycharmProjects/B2Q_code/dqn_allrings_moreatoms/",
                          top_k=5)

print()
summarise_molecule_output("/data/PycharmProjects/B2Q_code/dqnbs_allrings_moreatoms/",
                          top_k=5)

print()
summarise_molecule_output("/data/PycharmProjects/B2Q_code/bdqn_allrings_moreatoms/",
                          top_k=5)
