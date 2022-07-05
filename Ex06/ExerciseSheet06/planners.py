import torch
import numpy as np


class Planner:
    def predict(self, model, actions, observation):
        observations = []
        for t in range(self._horizon):
            if len(actions[t].shape) > 1:
                inp = torch.cat([observation, actions[t]], dim=1)
            else:
                inp = torch.cat([observation, actions[t]])
            observation = model.forward(inp)
            observations.append(observation)
        return observations

    def __call__(self, model, observation):
        pass


class RandomPlanner(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=20,
    ):
        Planner.__init__(self)
        self._horizon = horizon
        self._action_size = action_size

    def __call__(self, model, observation):
        actions = torch.rand([self._horizon, self._action_size]) * 2 - 1
        with torch.no_grad():
            observations = self.predict(model, actions, observation)
        return actions, observations


class CrossEntropyMethod(Planner):
    def __init__(
        self,
        action_size=2,
        horizon=50,
        num_inference_cycles=2,
        num_predictions=50,
        num_elites=5,
        num_keep_elites=2,
        criterion=torch.nn.MSELoss(),
        policy_handler=lambda x: x,
        var=0.2,
    ):
        Planner.__init__(self)
        self._action_size = action_size
        self._horizon = horizon
        self._num_inference_cycles = num_inference_cycles
        self._num_predictions = num_predictions
        self._num_elites = num_elites
        self._num_keep_elites = num_keep_elites
        self._criterion = criterion
        self._policy_handler = policy_handler

        self._mu = torch.zeros([self._horizon, self._action_size])
        self._var_init = var * torch.ones([self._horizon, self._action_size])
        self._var = self._var_init.clone().detach()
        self._dist = torch.distributions.MultivariateNormal(
            torch.zeros(self._action_size), torch.eye(self._action_size)
        )

        self._last_actions = None

    def __call__(self, model, observation):

        with torch.no_grad():
            for _ in range(self._num_inference_cycles):

                # DONE: implement CEM#
                # sampling
                actions = torch.zeros((self._horizon, self._num_predictions, self._action_size)) # [time, batch, action]
                for i in range(self._horizon):
                    actions[i] = self._dist.sample((self._num_predictions,)) * self._var[i] + self._mu[i]

                actions = actions.permute(1, 0, 2) # [batch, time, action]
                # actions = torch.cat([self._last_actions, actions])
                # TODO ask about last_actions

                actions = self._policy_handler(actions)

                # loss calculation
                losses = torch.zeros(self._num_predictions)
                for i in range(self._num_predictions):
                    observations = self.predict(model, actions[i], observation) # [50, 2]
                    losses[i] = self._criterion(torch.stack(observations)[:,None,:])

                # sorting and extracting indices
                indices = np.argsort(losses)
                sorted_actions = actions[indices]
                elites = sorted_actions[:self._num_elites]
                self._last_actions = sorted_actions[:self._num_keep_elites] # [batch, time, action] [2, 50, 2]

                # calc new 
                self._mu = torch.mean(elites, dim=0)
                self._var = torch.var(elites, dim=0)

                shape = (self._horizon, self._action_size)
                assert self._mu.shape == shape, f"Shape of mu should be {shape}- is {self._mu.shape}"
                assert self._var.shape == shape, f"Shape of var should be {shape} - is {self._var.shape}"

        # Policy has been optimized; this optimized policy is now propagated
        # once more in forward direction in order to generate the final
        # observations to be returned
        actions = actions.permute(1, 0, 2)  # [time, batch, action] [self._horizon, self.num_predictions  self._action_size]
        with torch.no_grad():
            observations = self.predict(model, actions[:, 0, :], observation)

        with torch.no_grad():
            # Shift means for one time step
            self._mu[:-1] = self._mu[1:].clone()
            # Reset the variance
            self._var = self._var_init.clone()
            # Shift elites to keep for one time step
            self._last_actions[:, :-1] = self._last_actions[:, 1:].clone()

        return actions[:, 0, :], observations
