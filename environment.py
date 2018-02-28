import numpy as np
class Workspace():
	def __init__(self, conversion_rate_distribution):
		self.conversion_rate = conversion_rate_distribution
	def get_reward(self, timestep):
		return self.conversion_rate(timestep)



class Environment():
	def __init__(self, workspaces):
		self.workspaces = workspaces

	def act(self, chosen_action, timestep):
		reward = int(self.workspaces[chosen_action].get_reward(timestep))

		return reward

