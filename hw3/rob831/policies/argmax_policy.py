import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## TODOX return the action that maximizes the Q-value
        # at the current observation as the output
        # q_values = self.critic.q_net(ptu.from_numpy(observation))
        # action = q_values.argmax(dim=-1)
        action = np.argmax(self.critic.qa_values(observation), axis=1)
        return action.squeeze()
