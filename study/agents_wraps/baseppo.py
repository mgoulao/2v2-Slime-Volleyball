import os

import torch

class BasePPO:
    def __init__(self):
        pass

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

class BaseTeam:
    def __init__(self, logdir):
        self.logdir = logdir

    @staticmethod
    def existsBestModel(logdir):
        return os.path.exists(f'{logdir}/best_model_agent1')

    def loadBestModel(self):
        bestSaveExists = os.path.exists(f'{self.logdir}/best_model_agent1')
        if bestSaveExists:
            print("TEAM 1: Best Model Loaded!")
            self.load("best_model")

    def saveBestModel(self):
        self.save("best_model")

    def save(self, filename):
        self.agent1.save(f'{self.logdir}/{filename}_agent1')
        self.agent2.save(f'{self.logdir}/{filename}_agent2')

    def load(self, filename):
        self.agent1.load(f'{self.logdir}/{filename}_agent1')
        self.agent2.load(f'{self.logdir}/{filename}_agent2')