#!/usr/bin/env python3

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, einsum

__author__ = "Matthias Karlbauer, Jannik Thümmel"


class FeedForward(nn.Module):
	"""
	A specific feed forward module that concists of a relu layer followed by a
	dropout and a linear layer.
	"""

	def __init__(self, d_model, linear_layer_size, dropout = 0.1):
		"""
		Constructor method of the feed forward module.
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()

		# DONE: define the feed-forward and an according dropout layer here.
		self.layer1 = th.nn.Linear(d_model, linear_layer_size)
		self.layer2 = th.nn.Linear(linear_layer_size, d_model)
		self.dropout_layer = th.nn.Dropout(dropout)

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""
		
		# DONE: implement the forward pass for the feed-forward module here
		x = self.layer1(x)
		x = th.nn.functional.relu(x)
		x = self.dropout_layer(x)
		x = self.layer2(x)
		return x


class MultiHeadSelfAttention(nn.Module):
	"""
	The core component of the transformer realizing the attention mechanism.
	"""

	def __init__(self, n_heads, d_model, dropout = 0.1):
		"""
		Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
        :param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.dropout = dropout

		# DONE: set up the layers for the multi-head-attention module here
		# DONE: implement dropout
		self.W_Q = th.nn.parameter.Parameter(th.ones((d_model+1, n_heads, int(self.d_model / n_heads))))  # .cuda() ?
		self.W_K = th.nn.parameter.Parameter(th.ones((d_model+1, n_heads, int(self.d_model / n_heads))))
		self.W_V = th.nn.parameter.Parameter(th.ones((d_model+1, n_heads, int(self.d_model / n_heads))))
		self.W_O = th.nn.parameter.Parameter(th.ones((d_model+1, d_model)))  # +1 to model biases
		#th.nn.init.xavier_uniform(self.W_Q)
		#th.nn.init.xavier_uniform(self.W_K)
		#th.nn.init.xavier_uniform(self.W_V)
		#th.nn.init.xavier_uniform(self.W_O)

		self.dropout_layer = th.nn.Dropout(dropout)

	def forward(self, x, mask=None):
		"""
		Forward pass of the multi head attention module.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: The attention weighted output as linear combination of v
		"""
		# DONE: define the forward pass of the multi-head-attention here
		x = th.cat((x, th.ones((x.shape[0],1)).to(x.device)), 1)# pad with a column of ones to model biases
		q = th.einsum('ij, jlm -> ilm', x, self.W_Q) # seq_len, n_heads, d_model / n_heads
		k = th.einsum('ij, jlm -> ilm', x, self.W_K)
		v = th.einsum('ij, jlm -> ilm', x, self.W_V)

		results = []
		for head in range(self.n_heads):
			att, _ = self.attention(q[:,head,:], k[:,head,:], v[:,head,:], mask=mask)  # seq_len, d_model / n_heads
			results.append(att)
		concatted = th.concat(results, dim=1)  # seq_len, d_model
		concatted = th.cat((concatted, th.ones((concatted.shape[0], 1)).to(concatted.device)), 1)  # pad with a column of ones to model biases
		res = concatted @ self.W_O
		return self.dropout_layer(res)

	def attention(self, q, k, v, mask=None):
		"""
		The attention mechanism computing a weighted linear combination of v
		based on the similarity of the according k and v entries.
		:param k: Key vector
		:param v: Value vector
		:param q: Query vector
		:param mask: Mask to hide future entries
		:return: Weighted linear combination v_hat and the attention weights
		"""
		# DONE: compute the attention scores, apply the mask and perform the
		#		attention multiplication here. Remember the scaling factor!

		I = q @ k.T
		I_masked = I / np.sqrt(self.d_model) + mask
		softm = th.nn.functional.softmax(I_masked, dim=1)
		return softm @ v, softm


class DecoderLayer(nn.Module):
	"""
	A decoder layer part (of a Transformer) which predicts next observations
	based on the previous inputs.
	"""

	def __init__(self, n_heads, d_model, linear_layer_size, dropout=0.1):
		""" Constructor method of the attention module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param dropout: Probability of dropping out certain neurons
		"""

		# DONE: define the layers multi-head-attention, feed-forward, dropout
		# and normalization layers here.
		# Note that we do not use an Encoder
		# and therefore do not require the Encoder-Decoder Attention module
		super().__init__()
		self.att_mod = MultiHeadSelfAttention(n_heads, d_model, dropout)
		self.norm1 = th.nn.LayerNorm(d_model)
		self.ff_mod = FeedForward(d_model, linear_layer_size, dropout)
		self.norm2 = th.nn.LayerNorm(d_model)
		self.dropout_layer = th.nn.Dropout(dropout)

	def forward(self, x, mask):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:param mask: Mask to hide future entries
		:return: The module's output
		"""
		
		# DONE: define the forward pass. Keep in mind to produce residuals
		#		instead of the absolute values directly.
		x = x + self.att_mod(x, mask)
		x = self.norm1(x)
		x = x + self.ff_mod(x)
		x = self.norm2(x)
		x = self.dropout_layer(x)
		return x


class Model(nn.Module):
	"""
	The actual model consisting of a selected number of sequential decoder
	layers.
	"""

	def __init__(self, n_heads, d_model, linear_layer_size, d_one_hot, num_blocks,
				 dropout=0.1):
		"""
		Constructor method of the Model module.
		:param n_heads: The number of attention heads
		:param d_model: The size of the K, V, Q and output vectors
		:param linear_layer_size: The internal size of the feed forward module
		:param d_one_hot: The size of the input and output vector
		:param num_blocks: How many Transformer blocks to stack.
		:param dropout: Probability of dropping out certain neurons
		"""
		super().__init__()
		self.d_model = d_model
		# DONE: define linear and decoder layers for the overall model
		self.lin_in = th.nn.Linear(d_one_hot, d_model)
		self.blocks = nn.ModuleList()
		for _ in range(num_blocks):
			self.blocks.append(DecoderLayer(n_heads, d_model, linear_layer_size, dropout))
		self.lin_out = th.nn.Linear(d_model, d_one_hot)
		self.softmax = th.nn.Softmax(dim=1)

	def forward(self, x):
		"""
		The forward pass function of the module.
		:param x: The input to the module
		:return: The module's output
		"""
		# DONE: implement the forward pass of the model here
		x = self.lin_in(x)
		for block in self.blocks:
			x = block(x, self._mask(x))
		x = self.lin_out(x)  # x has shape (sentence_length, alphabet)
		x = self.softmax(x)
		return x
		
	def _mask(self, x):
		"""
		Helper function to compute the mask applied to the decoder layer
		:param x: The input data that should be masked
		"""
		device, seq_len = x.device, x.shape[0]    

		# DONE: implement the mask for the decoder here
		mask = (-np.inf * th.ones(seq_len, seq_len)).triu(diagonal=1).to(device)
		#mask = th.zeros((seq_len, seq_len)).to(device)  # no mask
		return mask
