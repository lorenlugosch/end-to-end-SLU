import torch
import numpy as np
import sys
import os
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
		self.is_cuda = next(self.parameters()).is_cuda
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
		num_rnn_layers = len(config.phone_rnn_num_hidden)
		out_dim = config.cnn_N_filt[-1]
		for idx in range(num_rnn_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.phone_rnn_num_hidden[idx], batch_first=True, bidirectional=config.phone_rnn_bidirectional)
			layer.name = "phone_rnn%d" % idx
			self.phoneme_layers.append(layer)
		
			out_dim = config.phone_rnn_num_hidden[idx]
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
		num_rnn_layers = len(config.word_rnn_num_hidden)
		for idx in range(num_rnn_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.word_rnn_num_hidden[idx], batch_first=True, bidirectional=config.word_rnn_bidirectional)
			layer.name = "word_rnn%d" % idx
			self.word_layers.append(layer)
		
			out_dim = config.word_rnn_num_hidden[idx]
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
		self.pretraining_type = config.pretraining_type
		if self.is_cuda:
			self.cuda()

	def forward(self, x, y_phoneme, y_word):
		"""
		x : Tensor of shape (batch size, T)
		y_phoneme : LongTensor of shape (batch size, T')
		y_word : LongTensor of shape (batch size, T'')

		Compute loss for y_word and y_phoneme for each x in the batch.
		"""
		self.is_cuda = next(self.parameters()).is_cuda
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

		phoneme_loss = torch.nn.functional.cross_entropy(phoneme_logits, y_phoneme, ignore_index=-1)
		valid_phoneme_indices = y_phoneme!=-1
		phoneme_acc = (phoneme_logits.max(1)[1][valid_phoneme_indices] == y_phoneme[valid_phoneme_indices]).float().mean()

		# avoid computing 
		if self.pretraining_type == 1:
			word_loss = torch.tensor([0.])
			word_acc = torch.tensor([0.])
		else:
			for layer in self.word_layers:
				out = layer(out)
			word_logits = self.word_linear(out)
			word_logits = word_logits.view(word_logits.shape[0]*word_logits.shape[1], -1)
			y_word = y_word.view(-1)

			word_loss = torch.nn.functional.cross_entropy(word_logits, y_word, ignore_index=-1)
			valid_word_indices = y_word!=-1
			word_acc = (word_logits.max(1)[1][valid_word_indices] == y_word[valid_word_indices]).float().mean()

		return phoneme_loss, word_loss, phoneme_acc, word_acc

	def compute_posteriors(self, x):
		self.is_cuda = next(self.parameters()).is_cuda
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
		self.is_cuda = next(self.parameters()).is_cuda
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

def has_params(layer):
	num_params = sum([p.numel() for p in layer.parameters()])
	if num_params > 0: return True
	return False

def is_frozen(layer):
	for param in layer.parameters():
		if param.requires_grad: return False
	return True

class Seq2SeqEncoder(torch.nn.Module):
	def __init__(self, input_dim, num_layers, encoder_dim):
		super(Seq2SeqEncoder, self).__init__()
		out_dim = input_dim
		self.layers = []
		for idx in range(num_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=encoder_dim, batch_first=True, bidirectional=True)
			layer.name = "intent_encoder_rnn%d" % idx
			self.layers.append(layer)
		
			out_dim = encoder_dim
			out_dim *= 2 # bidirectional

			# grab hidden states of RNN for each timestep
			layer = RNNSelect()
			layer.name = "intent_encoder_rnn_select%d" % idx
			self.layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=0.5)
			layer.name = "intent_encoder_dropout%d" % idx
			self.layers.append(layer)

		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, x):
		out = x
		for layer in self.layers:
			out = layer(out)
		return out

class Attention(torch.nn.Module):
	def __init__(self, encoder_dim, decoder_dim, key_dim, value_dim):
		super(Attention, self).__init__()
		self.scale_factor = torch.sqrt(torch.tensor(key_dim).float())
		self.key_linear = torch.nn.Linear(encoder_dim, key_dim)
		self.query_linear = torch.nn.Linear(decoder_dim, key_dim)
		self.value_linear = torch.nn.Linear(encoder_dim, value_dim)
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, encoder_states, decoder_state):
		"""
		encoder_states: Tensor of shape (batch size, T, encoder_dim)
		decoder_state: Tensor of shape (batch size, decoder_dim)

		Map the input sequence to a summary vector (batch size, value_dim) using attention, given a query.
		"""
		keys = self.key_linear(encoder_states)
		values = self.value_linear(encoder_states)
		query = self.query_linear(decoder_state)
		query = query.unsqueeze(2)
		scores = torch.matmul(keys, query) / self.scale_factor
		normalized_scores = self.softmax(scores).transpose(1,2)
		out = torch.matmul(normalized_scores, values).squeeze(1)
		return out

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()
		# self.gru = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden)
		# self.dropout = torch.nn.Dropout(dropout)

		self.layers = []
		self.num_layers = num_decoder_layers
		for index in range(num_decoder_layers):
			if index == 0: 
				layer = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
			else:
				layer = torch.nn.GRUCell(input_size=num_decoder_hidden, hidden_size=num_decoder_hidden) 
			layer.name = "gru%d"%index
			self.layers.append(layer)

			layer = torch.nn.Dropout(p=dropout)
			layer.name = "dropout%d"%index
			self.layers.append(layer)
		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, input, previous_state):
		"""
		input: Tensor of shape (batch size, input_size)
		previous_state: Tensor of shape (batch size, num_decoder_layers, num_decoder_hidden)

		Given the input vector, update the hidden state of each decoder layer.
		"""
		# return self.gru(input, previous_state)

		state = []
		batch_size = input.shape[0]
		gru_index = 0
		for index, layer in enumerate(self.layers):
			if index == 0:
				layer_out = layer(input, previous_state[:, gru_index])
				state.append(layer_out)
				gru_index += 1
			else:
				if "gru" in layer.name:
					layer_out = layer(layer_out, previous_state[:, gru_index])
					state.append(layer_out)
					gru_index += 1
				else: 
					layer_out = layer(layer_out)
		state = torch.stack(state, dim=1)
		return state 

def sort_beam(beam_extensions, beam_extension_scores, beam_pointers):
	beam_width = len(beam_pointers); batch_size = beam_pointers[0].shape[0]
	beam_extensions = torch.stack(beam_extensions); beam_extension_scores = torch.stack(beam_extension_scores); beam_pointers = torch.stack(beam_pointers)
	beam_extension_scores = beam_extension_scores.view(beam_width,batch_size)

	sort_order = beam_extension_scores.sort(dim=0, descending=True)[1].reshape(beam_width, batch_size)
	sorted_beam_extensions = beam_extensions.clone(); sorted_beam_extension_scores = beam_extension_scores.clone(); sorted_beam_pointers = beam_pointers.clone()
	
	for batch_index in range(batch_size):
		sorted_beam_extensions[:, batch_index] = beam_extensions[sort_order[:,batch_index], batch_index]
		sorted_beam_extension_scores[:, batch_index] = beam_extension_scores[sort_order[:,batch_index], batch_index]
		sorted_beam_pointers[:, batch_index] = beam_pointers[sort_order[:,batch_index], batch_index]
	return sorted_beam_extensions, sorted_beam_extension_scores, sorted_beam_pointers

class Seq2SeqDecoder(torch.nn.Module):
	"""
	Attention-based decoder for seq2seq SLU
	"""
	def __init__(self, num_labels, num_layers, encoder_dim, decoder_dim, key_dim, value_dim, SOS=0):
		super(Seq2SeqDecoder, self).__init__()
		embedding_dim = decoder_dim
		self.embed = torch.nn.Linear(num_labels, embedding_dim)
		self.attention = Attention(encoder_dim*2, decoder_dim, key_dim, value_dim)
		self.rnn = DecoderRNN(num_layers, decoder_dim, embedding_dim + value_dim, dropout=0.5)
		self.initial_state = torch.nn.Parameter(torch.randn(num_layers,decoder_dim))
		self.linear = torch.nn.Linear(decoder_dim, num_labels)
		self.log_softmax = torch.nn.LogSoftmax(dim=1)
		self.SOS = SOS # index of SOS label

	def forward(self, encoder_outputs, y, y_lengths=None):
		"""
		encoder_outputs : Tensor of shape (batch size, T, encoder output dim)
		y : Tensor of shape (batch size, U, num_labels) - padded with end-of-sequence tokens
		y_lengths : list of integers
		Compute log p(y|x) for each (x,y) in the batch.
		"""
		#if self.is_cuda:
		#	x = x.cuda()
		#	y = y.cuda()
		self.is_cuda = next(self.parameters()).is_cuda

		batch_size = y.shape[0]
		U = y.shape[1]
		num_labels = y.shape[2]

		# Initialize the decoder state
		decoder_state = torch.stack([self.initial_state] * batch_size)

		# Initialize log p(y|x) to 0, y_u-1 to SOS
		log_p_y_x = 0
		y_u_1 = torch.zeros(batch_size, num_labels)
		y_u_1[:,self.SOS] = 1.
		if self.is_cuda: y_u_1 = y_u_1.cuda()
		for u in range(0, U):
			# Feed in the previous element of y and the attention output; update the decoder state
			context = self.attention(encoder_outputs, decoder_state[:,-1])
			embedding = self.embed(y_u_1)
			decoder_input = torch.cat([embedding, context], dim=1)
			decoder_state = self.rnn(decoder_input, decoder_state)

			# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
			decoder_out = self.log_softmax(self.linear(decoder_state[:,-1]))
			log_p_yu = (decoder_out * y[:,u,:]).sum(dim=1) # y_u is one-hot; use dot-product to select the y_u'th output probability 

			# Add log p(y_u|...) to log p(y|x)
			log_p_y_x += log_p_yu # TODO: mask based on y_lengths?

			# Look at next element of y
			y_u_1 = y[:,u,:]

		return log_p_y_x

	def infer(self, encoder_outputs, Sy, B=4, debug=False, y_lengths=None):
		"""
		encoder_outputs : Tensor of shape (batch size, T, encoder_dim*2)
		Sy : list of characters (output alphabet)
		B : integer (beam width)
		debug : boolean (print debugging statements during search)
		Run beam search to find y_hat = argmax_y log p(y|x) for every (x) in the batch.
		(If B = 1, this is equivalent to greedy search.)
		"""
		#if self.is_cuda: x = x.cuda()
		self.is_cuda = next(self.parameters()).is_cuda

		batch_size = encoder_outputs.shape[0]
		Sy_size = len(Sy)

		# Initialize the decoder state
		decoder_state = torch.stack([self.initial_state] * batch_size)

		true_U = 200

		if y_lengths is not None:
			true_U = max(y_lengths)

		decoder_state_shape = decoder_state.shape
		beam = torch.zeros(B,batch_size,true_U,Sy_size); beam_scores = torch.zeros(B,batch_size); decoder_states = torch.zeros(B,decoder_state_shape[0], decoder_state_shape[1], decoder_state_shape[2])
		if self.is_cuda:
			beam = beam.cuda()
			beam_scores = beam_scores.cuda()
			decoder_states = decoder_states.cuda()

		for u in range(true_U):
			beam_extensions = []; beam_extension_scores = []; beam_pointers = []

			# Add a delay so that it's easier to read the outputs during debugging
			if debug and u < true_U:
				time.sleep(1)
				print("")

			for b in range(B):
				# Get previous guess
				if u == 0: 
					beam_score = beam_scores[b]
					y_hat_u_1 = torch.zeros(batch_size, Sy_size)
					if self.is_cuda:
						beam_score = beam_score.cuda()
						y_hat_u_1 = y_hat_u_1.cuda()
				else: 
					# Select hypothesis (and corresponding decoder state/score) from beam
					y_hat = beam[b]
					decoder_state = decoder_states[b]
					beam_score = beam_scores[b]
					y_hat_u_1 = y_hat[:,u-1,:]

					# If in debug mode, print out the current beam
					if debug and u < true_U: print(self.one_hot_to_string(y_hat[0,:u], Sy).strip("\n") + " | score: %1.2f" % beam_score[0].item())

				# Feed in the previous guess; update the decoder state
				context = self.attention(encoder_outputs, decoder_state[:,-1])
				embedding = self.embed(y_hat_u_1)
				decoder_input = torch.cat([embedding, context], dim=1)
				decoder_state = self.rnn(decoder_input, decoder_state)
				decoder_states[b] = decoder_state.clone()

				# Compute log p(y_u|y_1, y_2, ..., x) (the log probability of the next element)
				decoder_out = self.log_softmax(self.linear(decoder_state[:,-1]))

				# Find the top B possible extensions for each of the B hypotheses
				top_B_extension_scores, top_B_extensions = decoder_out.topk(B)
				top_B_extension_scores = top_B_extension_scores.transpose(0,1); top_B_extensions = top_B_extensions.transpose(0,1)
				for extension_index in range(B):
					extension = torch.zeros(batch_size, Sy_size)
					extension_score = top_B_extension_scores[extension_index] + beam_score
					extension[torch.arange(batch_size), top_B_extensions[extension_index]] = 1.
					beam_extensions.append(extension.clone())
					beam_extension_scores.append(extension_score.clone())
					beam_pointers.append(torch.ones(batch_size).long() * b) # we need to remember which hypothesis this extension belongs to

				# At the first decoding timestep, there are no other hypotheses to extend.
				if u == 0: break

			# Sort the B^2 extensions
			beam_extensions, beam_extension_scores, beam_pointers = sort_beam(beam_extensions, beam_extension_scores, beam_pointers)
			old_beam = beam.clone(); old_beam_scores = beam_scores.clone(); old_decoder_states = decoder_states.clone()
			beam *= 0; beam_scores *= 0; decoder_states *= 0;

			# Pick the top B extended hypotheses
			for b in range(len(beam_extensions[:B])):
				for batch_index in range(batch_size):
					beam[b,batch_index] = old_beam[beam_pointers[b, batch_index],batch_index]
					beam[b,batch_index,u,:] = beam_extensions[b, batch_index] # append the extensions to each hypothesis
					beam_scores[b, batch_index] = beam_extension_scores[b, batch_index] # update the beam scores
					decoder_states[b, batch_index] = old_decoder_states[beam_pointers[b, batch_index],batch_index]

		return beam_scores, beam

class Model(torch.nn.Module):
	"""
	End-to-end SLU model.
	"""
	def __init__(self, config):
		super(Model, self).__init__()
		self.is_cuda = torch.cuda.is_available()
		self.Sy_intent = config.Sy_intent
		pretrained_model = PretrainedModel(config)
		if config.pretraining_type != 0:
			pretrained_model_path = os.path.join(config.folder, "pretraining", "model_state.pth")
			if self.is_cuda:
				pretrained_model.load_state_dict(torch.load(pretrained_model_path))
			else:
				pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
		self.pretrained_model = pretrained_model
		self.unfreezing_type = config.unfreezing_type
		self.unfreezing_index = config.starting_unfreezing_index
		self.intent_layers = []
		if config.pretraining_type != 0:
			self.freeze_all_layers()
		self.seq2seq = config.seq2seq
		out_dim = config.word_rnn_num_hidden[-1]
		if config.word_rnn_bidirectional:
			out_dim *= 2 

		# fixed-length output:
		if not self.seq2seq:
			self.values_per_slot = config.values_per_slot
			self.num_values_total = sum(self.values_per_slot)
			num_rnn_layers = len(config.intent_rnn_num_hidden)
			for idx in range(num_rnn_layers):
				# recurrent
				layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.intent_rnn_num_hidden[idx], batch_first=True, bidirectional=config.intent_rnn_bidirectional)
				layer.name = "intent_rnn%d" % idx
				self.intent_layers.append(layer)
		
				out_dim = config.intent_rnn_num_hidden[idx]
				if config.intent_rnn_bidirectional:
					out_dim *= 2

				# grab hidden states of RNN for each timestep
				layer = RNNSelect()
				layer.name = "intent_rnn_select%d" % idx
				self.intent_layers.append(layer)

				# dropout
				layer = torch.nn.Dropout(p=config.intent_rnn_drop[idx])
				layer.name = "intent_dropout%d" % idx
				self.intent_layers.append(layer)

				# downsample
				layer = Downsample(method=config.intent_downsample_type[idx], factor=config.intent_downsample_len[idx], axis=1)
				layer.name = "intent_downsample%d" % idx
				self.intent_layers.append(layer)

			layer = torch.nn.Linear(out_dim, self.num_values_total)
			layer.name = "final_classifier"
			self.intent_layers.append(layer)

			layer = FinalPool()
			layer.name = "final_pool"
			self.intent_layers.append(layer)

			self.intent_layers = torch.nn.ModuleList(self.intent_layers)

		# seq2seq
		else:
			self.SOS = config.Sy_intent.index("<sos>")
			#self.EOS = config.EOS
			self.num_labels = len(config.Sy_intent) 
			self.encoder = Seq2SeqEncoder(out_dim, config.num_intent_encoder_layers, config.intent_encoder_dim)
			self.decoder = Seq2SeqDecoder(self.num_labels, config.num_intent_decoder_layers, config.intent_encoder_dim, config.intent_decoder_dim, config.intent_decoder_key_dim, config.intent_decoder_value_dim, self.SOS)

		if self.is_cuda:
			self.cuda()

	def one_hot_to_string(self, input, S):
		"""
		input : Tensor of shape (T, |S|)
		S : list of characters/tokens
		"""

		return "".join([S[c] for c in input.max(dim=1)[1]]).lstrip("<sos>").rstrip("<eos>")

	def freeze_all_layers(self):
		for layer in self.pretrained_model.phoneme_layers:
			freeze_layer(layer)
		for layer in self.pretrained_model.word_layers:
			freeze_layer(layer)

	def print_frozen(self):
		for layer in self.pretrained_model.phoneme_layers:
			if has_params(layer):
				frozen = "frozen" if is_frozen(layer) else "unfrozen"
				print(layer.name + ": " + frozen)
		for layer in self.pretrained_model.word_layers:
			if has_params(layer):
				frozen = "frozen" if is_frozen(layer) else "unfrozen"
				print(layer.name + ": " + frozen)

	def unfreeze_one_layer(self):
		"""
		ULMFiT-style unfreezing:
			Unfreeze the next trainable layer
		"""
		# no unfreezing
		if self.unfreezing_type == 0:
			return

		if self.unfreezing_type == 1:
			trainable_index = 0 # which trainable layer
			global_index = 1 # which layer overall
			while global_index <= len(self.pretrained_model.word_layers):
				layer = self.pretrained_model.word_layers[-global_index]
				unfreeze_layer(layer)
				if has_params(layer): trainable_index += 1
				global_index += 1
				if trainable_index == self.unfreezing_index: 
					self.unfreezing_index += 1
					return

		if self.unfreezing_type == 2:
			trainable_index = 0 # which trainable layer
			global_index = 1 # which layer overall
			while global_index <= len(self.pretrained_model.word_layers):
				layer = self.pretrained_model.word_layers[-global_index]
				unfreeze_layer(layer)
				if has_params(layer): trainable_index += 1
				global_index += 1
				if trainable_index == self.unfreezing_index: 
					self.unfreezing_index += 1
					return

			global_index = 1
			while global_index <= len(self.pretrained_model.phoneme_layers):
				layer = self.pretrained_model.phoneme_layers[-global_index]
				unfreeze_layer(layer)
				if has_params(layer): trainable_index += 1
				global_index += 1
				if trainable_index == self.unfreezing_index:
					self.unfreezing_index += 1
					return

	def forward(self, x, y_intent):
		"""
		x : Tensor of shape (batch size, T)
		y_intent : LongTensor of shape (batch size, num_slots)
		"""
		if self.is_cuda:
			y_intent = y_intent.cuda()
		out = self.pretrained_model.compute_features(x)

		if not self.seq2seq:
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

		else: # seq2seq
			out = self.encoder(out)
			log_probs = self.decoder(out, y_intent)
			return -log_probs.mean(), torch.tensor([0.])

	def predict_intents(self, x):
		out = self.pretrained_model.compute_features(x)

		if not self.seq2seq:
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

		else: #seq2seq
			out = self.encoder(out)
			beam_scores, beam = self.decoder.infer(out, self.Sy_intent, B=4)
			return beam_scores, beam

	def decode_intents(self, x):
		_, predicted_intent = self.predict_intents(x)

		if not self.seq2seq:
			intents = []
			for prediction in predicted_intent:
				intent = []
				for idx, slot in enumerate(self.Sy_intent):
					for value in self.Sy_intent[slot]:
						if prediction[idx].item() == self.Sy_intent[slot][value]:
							intent.append(value)
				intents.append(intent)
			return intents

		else: # seq2seq
			intents = []
			#predicted_intent: (beam, batch, U, num_labels)
			batch_size = predicted_intent.shape[1]
			for i in range(0, batch_size): 
				intent = self.one_hot_to_string(predicted_intent[0,i],self.Sy_intent)
				intents.append(intent)
			return intents

