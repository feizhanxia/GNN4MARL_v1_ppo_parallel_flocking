# agents/policy_base.py

from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def act(self, obs):
        """
        Given environment observation, return actions.
        Expected return: torch.tensor of shape [N] (angle delta per agent)
        """
        pass

    def store_transition(self, *args, **kwargs):
        """
        Optional: for training algorithms that require memory (e.g. REINFORCE)
        """
        pass

    def update(self):
        """
        Optional: update the policy parameters after an episode or batch
        """
        pass
