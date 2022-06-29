#!/usr/bin/env python3

import torch as th
import torch.nn as nn

__author__ = "Matthias Karlbauer"


###########
# CLASSES #
###########

class Model(nn.Module):
	"""
	The actual model consisting of some feed forward and recurrent layers.
	"""

	def __init__(self, d_one_hot, d_lstm, num_lstm_layers, dropout=0.1,
				 bias=True):
		"""
		Constructor method of the Model module.
		:param d_one_hot: The size of the input and output vector
		:param d_lstm: The hidden size of the lstm layers
		:param num_lstm_layers: The number of sequential lstm layers
		:param dropout: Probability of dropping out certain neurons
		:param bias: Whether to use bias neurons
		"""
		super().__init__()

		self.input_layer = th.nn.Linear(d_one_hot, d_lstm, bias=bias)
		self.lstm_layer = th.nn.LSTM(num_layers=num_lstm_layers, hidden_size=d_lstm, input_size=d_lstm, dropout=dropout, bias=bias)
		self.output_layer = th.nn.Linear(d_lstm, d_one_hot, bias=bias)

	def forward(self, x, state=None):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""

		x = self.input_layer(x)
		# h0 = th.zeros(self.lstm_layer.num_layers,1,self.lstm_layer.hidden_size)
		# c0 = th.zeros(self.lstm_layer.num_layers,1,self.lstm_layer.hidden_size)
		if state:
			h0, c0 = state
			x, (h, c) = self.lstm_layer(x, (h0.detach(), c0.detach()))
		else:
			x, (h, c) = self.lstm_layer(x)
		x = self.output_layer(x)

		# DONE: Implement the forward pass and return the model's output as
		#		well as the hidden and cell states of the lstms.

		return x, (h, c)
