import torch
import numpy as np
import sys

import math

def flip(x, dim):
	xsize = x.size()
	dim = x.dim() + dim if dim < 0 else dim
	x = x.contiguous()
	x = x.view(-1, *xsize[dim:])
	x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
		-1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
	return x.view(xsize)


def sinc(band,t_right):
	y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
	y_left= flip(y_right,0)

	if t_right.is_cuda: y=torch.cat([y_left,(torch.ones(1)).cuda(),y_right])
	else: y=torch.cat([y_left,(torch.ones(1)),y_right])

	return y

class Downsample(torch.nn.Module):
	"""
	Downsamples the input in the time/sequence domain
	"""
	def __init__(self, method="none", factor=1, axis=1):
		super(Downsample,self).__init__()
		self.factor = factor
		self.method = method
		self.axis = axis
		methods = ["none", "avg", "max"]
		if self.method not in methods:
			print("Error: downsampling method must be one of the following: \"none\", \"avg\", \"max\"")
			sys.exit()
			
	def forward(self, x):
		if self.method == "none":
			return x.transpose(self.axis, 0)[::self.factor].transpose(self.axis, 0)
		if self.method == "avg":
			return torch.nn.functional.avg_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
		if self.method == "max":
			return torch.nn.functional.max_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
			

class SincLayer(torch.nn.Module):
	"""
	Modified from https://github.com/mravanelli/SincNet/blob/master/dnn_models.py:sinc_conv
	"""
	def __init__(self, N_filt,Filt_dim,fs, stride=1, padding=0, is_cuda=False):
		super(SincLayer,self).__init__()

		# Mel Initialization of the filterbanks
		low_freq_mel = 80
		high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
		mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
		f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
		b1=np.roll(f_cos,1)
		b2=np.roll(f_cos,-1)
		b1[0]=30
		b2[-1]=(fs/2)-100

		self.freq_scale=fs*1.0
		self.filt_b1 = torch.nn.Parameter(torch.from_numpy(b1/self.freq_scale))
		self.filt_band = torch.nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

		self.N_filt=N_filt
		self.Filt_dim=Filt_dim
		self.fs=fs
		self.stride=stride
		self.padding=padding
		self.is_cuda = is_cuda

	def forward(self, x):
		filters=torch.zeros((self.N_filt,self.Filt_dim)) #.cuda()
		if self.is_cuda: filters = filters.cuda()
		N=self.Filt_dim
		t_right=(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs) #.cuda()
		if self.is_cuda: t_right = t_right.cuda()

		min_freq=50.0;
		min_band=50.0;

		filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
		filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)

		n=torch.linspace(0, N, steps=N)

		# Filter window (hamming)
		window=0.54-0.46*torch.cos(2*math.pi*n/N);
		window=window.float() #.cuda()
		if self.is_cuda: window = window.cuda()

		for i in range(self.N_filt):
			low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
			low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
			band_pass=(low_pass2-low_pass1)

			band_pass=band_pass/torch.max(band_pass)
			if self.is_cuda: band_pass = band_pass.cuda()

			filters[i,:]=band_pass*window

			out=torch.nn.functional.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim), stride=self.stride, padding=self.padding)

		return out

class FinalPool(torch.nn.Module):
	def __init__(self):
		super(FinalPool, self).__init__()

	def forward(self, input):
		"""
		input : Tensor of shape (batch size, T, Cin)
		
		Outputs a Tensor of shape (batch size, Cin).
		"""

		return input.max(dim=1)[0]

class NCL2NLC(torch.nn.Module):
	def __init__(self):
		super(NCL2NLC, self).__init__()

	def forward(self, input):
		"""
		input : Tensor of shape (batch size, T, Cin)
		
		Outputs a Tensor of shape (batch size, Cin, T).
		"""

		return input.transpose(1,2)

class RNNSelect(torch.nn.Module):
	def __init__(self):
		super(RNNSelect, self).__init__()

	def forward(self, input):
		"""
		input : tuple of stuff
		
		Outputs a Tensor of shape 
		"""

		return input[0] 

class LayerNorm(torch.nn.Module):
	def __init__(self, dim, eps=1e-6):
		super(LayerNorm,self).__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		self.beta = nn.Parameter(torch.zeros(dim))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(1, keepdim=True)
		std = x.std(1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Abs(torch.nn.Module):
	def __init__(self):
		super(Abs, self).__init__()

	def forward(self, input):
		return torch.abs(input) 

class PretrainedModel(torch.nn.Module):
	"""
	Model pre-trained to recognize phonemes and words.
	"""
	def __init__(self, config):
		super(PretrainedModel, self).__init__()
		self.phoneme_layers = []
		self.word_layers = []
		self.is_cuda = torch.cuda.is_available()

		# CNN
		num_conv_layers = len(config.cnn_N_filt)
		for idx in range(num_conv_layers):
			# first conv layer
			if idx == 0:
				# if config.cnn_use_batchnorm_inp:
				# 	layer=torch.nn.BatchNorm1d([self.input_dim],momentum=0.05)
				# 	layer.name="bn0"
				# 	self.layers.append(layer)

				# if config.cnn_use_laynorm_inp:
				# 	layer=LayerNorm(self.input_dim)
				# 	layer.name="ln0"
				# 	self.layers.append(layer)

				if config.use_sincnet:
					layer = SincLayer(config.cnn_N_filt[idx], config.cnn_len_filt[idx], config.fs, stride=config.cnn_stride[idx], padding=config.cnn_len_filt[idx]//2, is_cuda=self.is_cuda)
					layer.name = "sinc%d" % idx
					self.phoneme_layers.append(layer)
				else:
					layer = torch.nn.Conv1d(1, config.cnn_N_filt[idx], config.cnn_len_filt[idx], stride=config.cnn_stride[idx], padding=config.cnn_len_filt[idx]//2)
					layer.name = "conv%d" % idx
					self.phoneme_layers.append(layer)

				layer = Abs()
				layer.name = "abs%d" % idx
				self.phoneme_layers.append(layer)

			# subsequent conv layers
			else:
				layer = torch.nn.Conv1d(config.cnn_N_filt[idx-1], config.cnn_N_filt[idx], config.cnn_len_filt[idx], stride=config.cnn_stride[idx], padding=config.cnn_len_filt[idx]//2)
				layer.name = "conv%d" % idx
				self.phoneme_layers.append(layer)

			# # batch norm
			# if config.cnn_use_batchnorm[idx]: 
			# 	layer = torch.nn.BatchNorm1d([self.input_dim],momentum=0.05)
			# 	layer.name = "bn%d" % idx
			# 	self.layers.append(layer)

			# # layer norm
			# if config.cnn_use_laynorm[idx]:
			# 	layer = LayerNorm()
			# 	layer.name = "ln%d" % idx
			# 	self.layers.append(layer)

			# pool
			layer = torch.nn.MaxPool1d(config.cnn_max_pool_len[idx], ceil_mode=True)
			layer.name = "pool%d" % idx
			self.phoneme_layers.append(layer)

			# activation
			if config.cnn_act[idx] == "leaky_relu":
				layer = torch.nn.LeakyReLU(0.2)
			else: 
				layer = torch.nn.ReLU()
			layer.name = "act%d" % idx
			self.phoneme_layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=config.cnn_drop[idx])
			layer.name = "dropout%d" % idx
			self.phoneme_layers.append(layer)

		# reshape output of CNN to be suitable for RNN (batch size, T, Cin)
		layer = NCL2NLC()
		layer.name = "ncl2nlc"
		self.phoneme_layers.append(layer)

		# phoneme RNN
		num_rnn_layers = len(config.phone_rnn_lay)
		out_dim = config.cnn_N_filt[-1]
		for idx in range(num_rnn_layers):
			# recurrent
			if config.phone_rnn_type == "gru":
				layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.phone_rnn_lay[idx], batch_first=True, bidirectional=config.phone_rnn_bidirectional)
			layer.name = "phone_rnn%d" % idx
			self.phoneme_layers.append(layer)
		
			out_dim = config.phone_rnn_lay[idx]
			if config.phone_rnn_bidirectional:
				out_dim *= 2

			# grab hidden states of RNN for each timestep
			layer = RNNSelect()
			layer.name = "phone_rnn_select%d" % idx
			self.phoneme_layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=config.phone_rnn_drop[idx])
			layer.name = "phone_dropout%d" % idx
			self.phoneme_layers.append(layer)

			# downsample
			layer = Downsample(method=config.phone_downsample_type[idx], factor=config.phone_downsample_len[idx], axis=1)
			layer.name = "phone_downsample%d" % idx
			self.phoneme_layers.append(layer)

		self.phoneme_layers = torch.nn.ModuleList(self.phoneme_layers)
		self.phoneme_linear = torch.nn.Linear(out_dim, config.num_phonemes)

		# word RNN
		num_rnn_layers = len(config.word_rnn_lay)
		for idx in range(num_rnn_layers):
			# recurrent
			if config.word_rnn_type == "gru":
				layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.word_rnn_lay[idx], batch_first=True, bidirectional=config.word_rnn_bidirectional)
			layer.name = "word_rnn%d" % idx
			self.word_layers.append(layer)
		
			out_dim = config.word_rnn_lay[idx]
			if config.word_rnn_bidirectional:
				out_dim *= 2

			# grab hidden states of RNN for each timestep
			layer = RNNSelect()
			layer.name = "word_rnn_select%d" % idx
			self.word_layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=config.word_rnn_drop[idx])
			layer.name = "word_dropout%d" % idx
			self.word_layers.append(layer)

			# downsample
			layer = Downsample(method=config.word_downsample_type[idx], factor=config.word_downsample_len[idx], axis=1)
			layer.name = "word_downsample%d" % idx
			self.word_layers.append(layer)

		self.word_layers = torch.nn.ModuleList(self.word_layers)
		self.word_linear = torch.nn.Linear(out_dim, config.vocabulary_size)
		if self.is_cuda:
			self.cuda()

	def forward(self, x, y_phoneme, y_word):
		"""
		x : Tensor of shape (batch size, T)
		y_phoneme : LongTensor of shape (batch size, T')
		y_word : LongTensor of shape (batch size, T'')

		Compute loss for y_word and y_phoneme for each x in the batch.
		"""
		if self.is_cuda:
			x = x.cuda()
			y_phoneme = y_phoneme.cuda()
			y_word = y_word.cuda()

		out = x.unsqueeze(1)
		for layer in self.phoneme_layers:
			out = layer(out)
		phoneme_logits = self.phoneme_linear(out)
		phoneme_logits = phoneme_logits.view(phoneme_logits.shape[0]*phoneme_logits.shape[1], -1)
		y_phoneme = y_phoneme.view(-1)

		for layer in self.word_layers:
			out = layer(out)
		word_logits = self.word_linear(out)
		word_logits = word_logits.view(word_logits.shape[0]*word_logits.shape[1], -1)
		y_word = y_word.view(-1)

		phoneme_loss = torch.nn.functional.cross_entropy(phoneme_logits, y_phoneme, ignore_index=-1)
		word_loss = torch.nn.functional.cross_entropy(word_logits, y_word, ignore_index=-1)

		phoneme_acc = (phoneme_logits.max(1)[1] == y_phoneme).float().mean()
		word_acc = (word_logits.max(1)[1] == y_word).float().mean()

		return phoneme_loss, word_loss, phoneme_acc, word_acc

	def compute_posteriors(self, x):
		if self.is_cuda:
			x = x.cuda()

		out = x.unsqueeze(1)
		for layer in self.phoneme_layers:
			out = layer(out)
		phoneme_logits = self.phoneme_linear(out)

		for layer in self.word_layers:
			out = layer(out)
		word_logits = self.word_linear(out)

		return phoneme_logits, word_logits

	def compute_features(self, x):
		if self.is_cuda:
			x = x.cuda()

		out = x.unsqueeze(1)
		for layer in self.phoneme_layers:
			out = layer(out)

		for layer in self.word_layers:
			out = layer(out)

		return out

def freeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = False

def unfreeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = True

class Model(torch.nn.Module):
	"""
	End-to-end SLU model.
	"""
	def __init__(self, config, pretrained_model):
		super(Model, self).__init__()
		self.pretrained_model = pretrained_model
		for layer in pretrained_model.phoneme_layers:
			freeze_layer(layer)
		for layer in pretrained_model.word_layers:
			freeze_layer(layer)
		self.intent_layers = []
		self.values_per_slot = config.values_per_slot
		self.num_values_total = sum(self.values_per_slot)

		out_dim = config.word_rnn_lay[-1]
		if config.word_rnn_bidirectional:
			out_dim *= 2
		layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.encoder_state_dim, batch_first=True, bidirectional=config.encoder_bidirectional)
		layer.name = "intent_rnn"
		self.intent_layers.append(layer)

		# grab hidden states of RNN for each timestep
		layer = RNNSelect()
		layer.name = "intent_rnn_select"
		self.intent_layers.append(layer)

		layer = torch.nn.Dropout(p=config.encoder_drop)
		layer.name = "intent_dropout"
		self.intent_layers.append(layer)

		out_dim = config.encoder_state_dim
		if config.encoder_bidirectional:
			out_dim *= 2
		layer = torch.nn.Linear(out_dim, self.num_values_total)
		layer.name = "final_classifier"
		self.intent_layers.append(layer)

		layer = FinalPool()
		layer.name = "final_pool"
		self.intent_layers.append(layer)

		self.intent_layers = torch.nn.ModuleList(self.intent_layers)

		self.is_cuda = torch.cuda.is_available()
		if self.is_cuda:
			self.cuda()

	def forward(self, x, y_intent):
		"""
		x : Tensor of shape (batch size, T)
		y_intent : LongTensor of shape (batch size, num_slots)
		"""
		if self.is_cuda:
			y_intent = y_intent.cuda()
		out = self.pretrained_model.compute_features(x)

		for layer in self.intent_layers:
			out = layer(out)
		intent_logits = out # shape: (batch size, num_values_total)

		intent_loss = 0.
		start_idx = 0
		predicted_intent = []
		for slot in range(len(self.values_per_slot)):
			end_idx = start_idx + self.values_per_slot[slot]
			subset = intent_logits[:, start_idx:end_idx]
			intent_loss += torch.nn.functional.cross_entropy(subset, y_intent[:, slot])
			predicted_intent.append(subset.max(1)[1])
			start_idx = end_idx
		predicted_intent = torch.stack(predicted_intent, dim=1)
		intent_acc = (predicted_intent == y_intent).prod(1).float().mean() # all slots must be correct

		return intent_loss, intent_acc

	def predict_intents(self, x):
		out = self.pretrained_model.compute_features(x)

		for layer in self.intent_layers:
			out = layer(out)
		intent_logits = out # shape: (batch size, num_values_total)

		start_idx = 0
		predicted_intent = []
		for slot in range(len(self.values_per_slot)):
			end_idx = start_idx + self.values_per_slot[slot]
			subset = intent_logits[:, start_idx:end_idx]
			predicted_intent.append(subset.max(1)[1])
			start_idx = end_idx
		predicted_intent = torch.stack(predicted_intent, dim=1)

		return intent_logits, predicted_intent
