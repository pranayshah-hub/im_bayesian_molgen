# import collections
import os


class MoleculeMonitor:
    def __init__(self,
                 folder):
        self.folder = folder

        # self.molecule_to_reward = collections.defaultdict(list)
        self.molecule_to_reward = dict()

    def add_molecule(self,
                     state,
                     reward):
        self.molecule_to_reward[state] = reward
        # if state in self.molecule_to_reward.keys():
        #     stored_reward = self.molecule_to_reward[state]
        #     if reward != stored_reward:
        #         raise ValueError("Now that the reward itself is undiscounted, this should not be reachable.")
        # else:
        #     self.molecule_to_reward[state] = reward

    def store_molecules(self):
        #TODO: This is not "thread-safe": i.e., if running 1 experiment per GPU in parallel, it may overwrite.
        counter = 0
        while os.path.exists(self.folder + "/" + "molecule_output" + "_trial" + repr(counter) + ".txt"):
            counter += 1

        with open(self.folder + "/" + "molecule_output" + "_trial" + repr(counter) + ".txt", "w") as fp:
            for k, v in self.molecule_to_reward.items():
                fp.write(k + "\t" + str(v) + "\n")
