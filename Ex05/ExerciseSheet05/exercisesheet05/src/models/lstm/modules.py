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

		# TODO: Set up a linear input linear layer, two lstm layers, and an
		#		output linear layer here

	def forward(self, x, state=None):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""

		# TODO: Implement the forward pass and return the model's output as
		#		well as the hidden and cell states of the lstms.

		return x, (h, c)
