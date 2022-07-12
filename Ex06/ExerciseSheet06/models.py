import torch

class SimpleModel:

    def __init__(self, env):
        self.agent_pos_upper_limits = torch.tensor(env.agent_pos_upper_limits).float()
        self.agent_pos_lower_limits = torch.tensor(env.agent_pos_lower_limits).float()
        # Scaling of action effect
        self.action_factor = env.action_factor

    def clip_positions(self, agent_pos):
        agent_pos = torch.where(agent_pos < self.agent_pos_lower_limits, self.agent_pos_lower_limits, agent_pos)
        agent_pos = torch.where(agent_pos > self.agent_pos_upper_limits, self.agent_pos_upper_limits, agent_pos)
        return agent_pos

    def forward(self, x):
        x = x.float()
        with torch.no_grad():
            agent_pos = x[..., :2]
            actions = x[..., 2:]
            agent_pos = agent_pos + self.action_factor * actions
            agent_pos = self.clip_positions(agent_pos)
        return agent_pos


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkModel, self).__init__()

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()



    def forward(self, x):
        
        torch.div(x, torch.Tensor([1.5, 1.5, 5, 5, 3.14, 5, 1, 1]))

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x
        
