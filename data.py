import torch
import torch.utils.data
import torchaudio
import os, glob
from collections import Counter
import soundfile as sf
import numpy as np
import configparser
import textgrid
import multiprocessing
import json
import pandas as pd
from subprocess import call

class Config:
	def __init__(self):
		self.use_sincnet = True

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")
	
	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "pretraining"))
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

	#[phoneme_module]
	config.use_sincnet=(parser.get("phoneme_module", "use_sincnet") == "True")
	config.fs=int(parser.get("phoneme_module", "fs"))

	config.cnn_N_filt=[int(x) for x in parser.get("phoneme_module", "cnn_N_filt").split(",")]
	config.cnn_len_filt=[int(x) for x in parser.get("phoneme_module", "cnn_len_filt").split(",")]
	config.cnn_stride=[int(x) for x in parser.get("phoneme_module", "cnn_stride").split(",")]
	config.cnn_max_pool_len=[int(x) for x in parser.get("phoneme_module", "cnn_max_pool_len").split(",")]
	config.cnn_act=[x for x in parser.get("phoneme_module", "cnn_act").split(",")]
	config.cnn_drop=[float(x) for x in parser.get("phoneme_module", "cnn_drop").split(",")]

	config.phone_rnn_num_hidden=[int(x) for x in parser.get("phoneme_module", "phone_rnn_num_hidden").split(",")]
	config.phone_downsample_len=[int(x) for x in parser.get("phoneme_module", "phone_downsample_len").split(",")]
	config.phone_downsample_type=[x for x in parser.get("phoneme_module", "phone_downsample_type").split(",")]
	config.phone_rnn_drop=[float(x) for x in parser.get("phoneme_module", "phone_rnn_drop").split(",")]
	config.phone_rnn_bidirectional=(parser.get("phoneme_module", "phone_rnn_bidirectional") == "True")

	#[word_module]
	config.word_rnn_num_hidden=[int(x) for x in parser.get("word_module", "word_rnn_num_hidden").split(",")]
	config.word_downsample_len=[int(x) for x in parser.get("word_module", "word_downsample_len").split(",")]
	config.word_downsample_type=[x for x in parser.get("word_module", "word_downsample_type").split(",")]
	config.word_rnn_drop=[float(x) for x in parser.get("word_module", "word_rnn_drop").split(",")]
	config.word_rnn_bidirectional=(parser.get("word_module", "word_rnn_bidirectional") == "True")
	config.vocabulary_size=int(parser.get("word_module", "vocabulary_size"))

	#[intent_module]
	config.intent_rnn_num_hidden=[int(x) for x in parser.get("intent_module", "intent_rnn_num_hidden").split(",")]
	config.intent_downsample_len=[int(x) for x in parser.get("intent_module", "intent_downsample_len").split(",")]
	config.intent_downsample_type=[x for x in parser.get("intent_module", "intent_downsample_type").split(",")]
	config.intent_rnn_drop=[float(x) for x in parser.get("intent_module", "intent_rnn_drop").split(",")]
	config.intent_rnn_bidirectional=(parser.get("intent_module", "intent_rnn_bidirectional") == "True")
	try:
		config.intent_encoder_dim=int(parser.get("intent_module", "intent_encoder_dim"))
		config.num_intent_encoder_layers=int(parser.get("intent_module", "num_intent_encoder_layers"))
		config.intent_decoder_dim=int(parser.get("intent_module", "intent_decoder_dim"))
		config.num_intent_decoder_layers=int(parser.get("intent_module", "num_intent_decoder_layers"))
		config.intent_decoder_key_dim=int(parser.get("intent_module", "intent_decoder_key_dim"))
		config.intent_decoder_value_dim=int(parser.get("intent_module", "intent_decoder_value_dim"))
	except:
		print("no seq2seq hyperparameters")

	#[pretraining]
	config.asr_path=parser.get("pretraining", "asr_path")
	config.pretraining_type=int(parser.get("pretraining", "pretraining_type")) # 0 - no pre-training, 1 - phoneme pre-training, 2 - phoneme + word pre-training, 3 - word pre-training
	if config.pretraining_type == 0: config.starting_unfreezing_index = 1 + len(config.word_rnn_num_hidden) + len(config.phone_rnn_num_hidden) + len(config.cnn_N_filt)
	if config.pretraining_type == 1: config.starting_unfreezing_index = 1 + len(config.word_rnn_num_hidden)
	if config.pretraining_type == 2: config.starting_unfreezing_index = 1
	if config.pretraining_type == 3: config.starting_unfreezing_index = 1
	config.pretraining_lr=float(parser.get("pretraining", "pretraining_lr"))
	config.pretraining_batch_size=int(parser.get("pretraining", "pretraining_batch_size"))
	config.pretraining_num_epochs=int(parser.get("pretraining", "pretraining_num_epochs"))
	config.pretraining_length_mean=float(parser.get("pretraining", "pretraining_length_mean"))
	config.pretraining_length_var=float(parser.get("pretraining", "pretraining_length_var"))

	#[training]
	config.slu_path=parser.get("training", "slu_path")
	config.unfreezing_type=int(parser.get("training", "unfreezing_type"))
	config.training_lr=float(parser.get("training", "training_lr"))
	config.training_batch_size=int(parser.get("training", "training_batch_size"))
	config.training_num_epochs=int(parser.get("training", "training_num_epochs"))
	config.real_dataset_subset_percentage=float(parser.get("training", "real_dataset_subset_percentage"))
	config.synthetic_dataset_subset_percentage=float(parser.get("training", "synthetic_dataset_subset_percentage"))
	config.real_speaker_subset_percentage=float(parser.get("training", "real_speaker_subset_percentage"))
	config.synthetic_speaker_subset_percentage=float(parser.get("training", "synthetic_speaker_subset_percentage"))
	config.train_wording_path=parser.get("training", "train_wording_path")
	if config.train_wording_path=="None": config.train_wording_path = None
	config.test_wording_path=parser.get("training", "test_wording_path")
	if config.test_wording_path=="None": config.test_wording_path = None
	try:
		config.augment = (parser.get("training", "augment")  == "True")
	except:
		# old config file with no augmentation
		config.augment = False

	try:
		config.seq2seq = (parser.get("training", "seq2seq")  == "True")
	except:
		# old config file with no seq2seq
		config.seq2seq = False

	try:
		config.dataset_upsample_factor = int(parser.get("training", "dataset_upsample_factor"))
	except:
		# old config file
		config.dataset_upsample_factor = 1

	# compute downsample factor (divide T by this number)
	config.phone_downsample_factor = 1
	for factor in config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len:
		config.phone_downsample_factor *= factor

	config.word_downsample_factor = 1
	for factor in config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len + config.word_downsample_len:
		config.word_downsample_factor *= factor

	return config

def get_SLU_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	base_path = config.slu_path

	# Split
	if not config.seq2seq:
		synthetic_train_df = pd.read_csv(os.path.join(base_path, "data", "synthetic_data.csv"))
		real_train_df = pd.read_csv(os.path.join(base_path, "data", "train_data.csv"))
		if "\"Unnamed: 0\"" in list(real_train_df): real_train_df = real_train_df.drop(columns="Unnamed: 0")
	else:
		synthetic_train_df = pd.read_csv(os.path.join(base_path, "data", "synthetic_data_seq2seq.csv"))
		real_train_df = pd.read_csv(os.path.join(base_path, "data", "train_data_seq2seq.csv"))
		if "\"Unnamed: 0\"" in list(real_train_df): real_train_df = real_train_df.drop(columns="Unnamed: 0")

	# Select random subset of speakers
	# First, check if "speakerId" is in the df columns
	if "speakerId" in list(real_train_df) and "speakerId" in list(synthetic_train_df):
		if config.real_speaker_subset_percentage < 1:
			speakers = np.array(list(Counter(real_train_df.speakerId)))
			np.random.shuffle(speakers)
			selected_speaker_count = round(config.real_speaker_subset_percentage * len(speakers))
			selected_speakers = speakers[:selected_speaker_count]
			real_train_df = real_train_df[real_train_df["speakerId"].isin(selected_speakers)]
		if config.synthetic_speaker_subset_percentage < 1:
			speakers = np.array(list(Counter(synthetic_train_df.speakerId)))
			np.random.shuffle(speakers)
			selected_speaker_count = round(config.synthetic_speaker_subset_percentage * len(speakers))
			selected_speakers = speakers[:selected_speaker_count]
			synthetic_train_df = synthetic_train_df[synthetic_train_df["speakerId"].isin(selected_speakers)]
	else:
		if "speakerId" in list(real_train_df): real_train_df = real_train_df.drop(columns="speakerId")
		if "speakerId" in list(synthetic_train_df): synthetic_train_df = synthetic_train_df.drop(columns="speakerId")
		if config.real_speaker_subset_percentage < 1:
			print("no speaker id listed in dataset .csv; ignoring speaker subset selection")
		if config.synthetic_speaker_subset_percentage < 1:
			print("no speaker id listed in dataset .csv; ignoring speaker subset selection")

	# Select random subset of training data
	if config.real_dataset_subset_percentage < 1:
		subset_size = round(config.real_dataset_subset_percentage * len(real_train_df))
		real_train_df = real_train_df.loc[np.random.choice(len(real_train_df), subset_size, replace=False)]
		#real_train_df = real_train_df.set_index(np.arange(len(real_train_df)))
	if config.synthetic_dataset_subset_percentage < 1:
		subset_size = round(config.synthetic_dataset_subset_percentage * len(synthetic_train_df))
		synthetic_train_df = synthetic_train_df.loc[np.random.choice(len(synthetic_train_df), subset_size, replace=False)]
		#synthetic_train_df = synthetic_train_df.set_index(np.arange(len(synthetic_train_df)))

	train_df = pd.concat([synthetic_train_df, real_train_df]).reset_index()
	if not config.seq2seq:
		valid_df = pd.read_csv(os.path.join(base_path, "data", "valid_data.csv"))
		test_df = pd.read_csv(os.path.join(base_path, "data", "test_data.csv"))
	else:
		valid_df = pd.read_csv(os.path.join(base_path, "data", "valid_data_seq2seq.csv"))
		test_df = pd.read_csv(os.path.join(base_path, "data", "test_data_seq2seq.csv"))

	if not config.seq2seq:
		# Get list of slots
		Sy_intent = {"action": {}, "object": {}, "location": {}}

		values_per_slot = []
		for slot in ["action", "object", "location"]:
			slot_values = Counter(train_df[slot])
			for idx,value in enumerate(slot_values):
				Sy_intent[slot][value] = idx
			values_per_slot.append(len(slot_values))
		config.values_per_slot = values_per_slot
		config.Sy_intent = Sy_intent
	else: #seq2seq
		import string
		all_chars = "".join(train_df.loc[i]["semantics"] for i in range(len(train_df))) + string.printable # all printable chars; TODO: unicode?
		all_chars = list(set(all_chars))
		Sy_intent = ["<sos>"]
		Sy_intent += all_chars
		Sy_intent.append("<eos>")
		config.Sy_intent = Sy_intent

	# If certain phrases are specified, only use those phrases
	if config.train_wording_path is not None:
		with open(config.train_wording_path, "r") as f:
			train_wordings = [line.strip() for line in f.readlines()]
		train_df = train_df.loc[train_df.transcription.isin(train_wordings)]
		train_df = train_df.set_index(np.arange(len(train_df)))

	if config.test_wording_path is not None:
		with open(config.test_wording_path, "r") as f:
			test_wordings = [line.strip() for line in f.readlines()]
		valid_df = valid_df.loc[valid_df.transcription.isin(test_wordings)]
		valid_df = valid_df.set_index(np.arange(len(valid_df)))
		test_df = test_df.loc[test_df.transcription.isin(test_wordings)]
		test_df = test_df.set_index(np.arange(len(test_df)))

	# Get number of phonemes
	if os.path.isfile(os.path.join(config.folder, "pretraining", "phonemes.txt")):
		Sy_phoneme = []
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "r") as f:
			for line in f.readlines():
				if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
		config.num_phonemes = len(Sy_phoneme)
	else:
		print("No phoneme file found.")

	# Create dataset objects
	train_dataset = SLUDataset(train_df, base_path, Sy_intent, config,upsample_factor=config.dataset_upsample_factor)
	valid_dataset = SLUDataset(valid_df, base_path, Sy_intent, config)
	test_dataset = SLUDataset(test_df, base_path, Sy_intent, config)

	return train_dataset, valid_dataset, test_dataset

# taken from https://github.com/jfsantos/maracas/blob/master/maracas/maracas.py
def rms_energy(x):
	return 10*np.log10((1e-12 + x.dot(x))/len(x))

class SLUDataset(torch.utils.data.Dataset):
	def __init__(self, df, base_path, Sy_intent, config, upsample_factor=1):
		"""
		df:
		Sy_intent: Dictionary (transcript --> slot values)
		config: Config object (contains info about model and training)
		"""
		self.df = df
		self.base_path = base_path
		self.Sy_intent = Sy_intent
		self.upsample_factor = upsample_factor
		self.augment = False #augment
		self.SNRs = [0,5,10,15,20]
		self.seq2seq = config.seq2seq

		self.loader = torch.utils.data.DataLoader(self, batch_size=config.training_batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsSLU(self.Sy_intent, self.seq2seq))

	def __len__(self):
		#if self.augment: return len(self.df)*2 # second half of dataset is augmented
		return len(self.df) * self.upsample_factor

	def __getitem__(self, idx):
		#augment = ((idx / len(self.df)) > 1) and self.augment
		#true_idx = idx
		idx = idx % len(self.df)

		wav_path = os.path.join(self.base_path, self.df.loc[idx].path)
		effect = torchaudio.sox_effects.SoxEffectsChain()
		effect.set_input_file(wav_path)

		augment = False
		if augment:
			# speed/tempo
			min_speed = 0.9; max_speed = 1.1; speed_range = max_speed-min_speed
			speed = speed_range * np.random.rand(1)[0] + min_speed
			effect.append_effect_to_chain("tempo", speed)
			del speed

			# volume
			min_gain = -10; max_gain = 10; gain_range = max_gain-min_gain
			gain_dB = gain_range * np.random.rand(1)[0] + min_gain
			gain = 10**(gain_dB/20)
			effect.append_effect_to_chain("vol", gain)
			del gain_dB


		wav, fs = effect.sox_build_flow_effects()
		x = wav[0].numpy()
		del wav, effect

		if augment:
			# crop
			min_length = round(x.shape[0]*0.9); max_length = round(x.shape[0]*1.1); length_range=max_length-min_length
			length = int(length_range * np.random.rand(1)[0] + min_length)
			start = int((x.shape[0]-length)/2)
			if start < 0:
				left_padding = -start
				right_padding = length-(x.shape[0]-start)
				x = np.pad(x,(left_padding, right_padding),mode="constant")
			else:
				start += np.random.randint(low=-start, high=1, size=1)[0]
				x = x[start:start+length]

			# noise (taken from https://github.com/jfsantos/maracas/blob/master/maracas/maracas.py)
			snr = np.random.choice(self.SNRs, 1, p=[1/len(self.SNRs) for _ in range(len(self.SNRs))])[0]
			noise = np.random.randn(len(x))
			N_dB = rms_energy(noise)
			S_dB = rms_energy(x)
			N_new = S_dB - snr
			noise_scaled = 10**(N_new/20) * noise / 10**(N_dB/20)
			x = x + noise_scaled

		if not self.seq2seq:
			y_intent = [] 
			for slot in ["action", "object", "location"]:
				value = self.df.loc[idx][slot]
				y_intent.append(self.Sy_intent[slot][value])
		else:
			# need sos, eos
			y_intent = [self.Sy_intent.index("<sos>")]
			y_intent += [self.Sy_intent.index(c) for c in self.df.loc[idx]["semantics"]]
			y_intent.append(self.Sy_intent.index("<eos>"))

		return (x, y_intent)

def one_hot(letters, S):
	"""
	letters : LongTensor of shape (batch size, sequence length)
	S : integer
	Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible letters).
	"""

	out = torch.zeros(letters.shape[0], letters.shape[1], S)
	for i in range(0, letters.shape[0]):
		for t in range(0, letters.shape[1]):
			out[i, t, letters[i,t]] = 1
	return out

class CollateWavsSLU:
	def __init__(self, Sy_intent, seq2seq):
		self.Sy_intent = Sy_intent
		self.num_labels = len(self.Sy_intent)
		self.seq2seq = seq2seq
		if self.seq2seq:
			self.EOS = self.Sy_intent.index("<eos>")

	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, intent labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y_intent = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_intent_ = batch[index]

			x.append(torch.tensor(x_).float())
			y_intent.append(torch.tensor(y_intent_).long())

		# pad all sequences to have same length
		if not self.seq2seq:
			T = max([len(x_) for x_ in x])
			for index in range(batch_size):
				x_pad_length = (T - len(x[index]))
				x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			x = torch.stack(x)
			y_intent = torch.stack(y_intent)

			return (x,y_intent)

		else: # seq2seq
			T = max([len(x_) for x_ in x])
			U = max([len(y_intent_) for y_intent_ in y_intent])
			for index in range(batch_size):
				x_pad_length = (T - len(x[index]))
				x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))
				y_pad_length = (U - len(y_intent[index]))
				y_intent[index] = torch.nn.functional.pad(y_intent[index], (0,y_pad_length), value=self.EOS)

			x = torch.stack(x)
			y_intent = torch.stack(y_intent)
			y_intent = one_hot(y_intent, self.num_labels)

			return (x,y_intent)

def get_ASR_datasets(config):
	"""
		Assumes that the data directory contains the following two directories:
			"audio" : wav files (split into train-clean, train-other, ...)
			"text" : alignments for each wav

	config: Config object (contains info about model and training)
	"""
	base_path = config.asr_path

	# Get only files with a label
	train_textgrid_paths = glob.glob(base_path + "/text/train*/*/*/*.TextGrid")
	train_wav_paths = [path.replace("text", "audio").replace(".TextGrid", ".wav") for path in train_textgrid_paths]
	valid_textgrid_paths = glob.glob(base_path + "/text/dev*/*/*/*.TextGrid")
	valid_wav_paths = [path.replace("text", "audio").replace(".TextGrid", ".wav") for path in valid_textgrid_paths]
	test_textgrid_paths = glob.glob(base_path + "/text/test*/*/*/*.TextGrid")
	test_wav_paths = [path.replace("text", "audio").replace(".TextGrid", ".wav") for path in test_textgrid_paths]
	
	# Get list of phonemes and words
	if os.path.isfile(os.path.join(config.folder, "pretraining", "phonemes.txt")) and os.path.isfile(os.path.join(config.folder, "pretraining", "words.txt")):
		Sy_phoneme = []
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "r") as f:
			for line in f.readlines():
				if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
		config.num_phonemes = len(Sy_phoneme)

		Sy_word = []
		with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
			for line in f.readlines():
				Sy_word.append(line.rstrip("\n"))

	else:
		print("Getting vocabulary...")
		phoneme_counter = Counter()
		word_counter = Counter()
		for path in valid_textgrid_paths:
			tg = textgrid.TextGrid()
			tg.read(path)
			phoneme_counter.update([phone.mark.rstrip("0123456789") for phone in tg.getList("phones")[0] if phone.mark != ''])
			word_counter.update([word.mark for word in tg.getList("words")[0]]) #if word.mark != ''])

		Sy_phoneme = list(phoneme_counter)
		Sy_word = [w[0] for w in word_counter.most_common(config.vocabulary_size)]
		config.num_phonemes = len(Sy_phoneme)
		with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "w") as f:
			for phoneme in Sy_phoneme:
				f.write(phoneme + "\n")

		with open(os.path.join(config.folder, "pretraining", "words.txt"), "w") as f:
			for word in Sy_word:
				f.write(word + "\n")

	print("Done.")

	# Create dataset objects
	train_dataset = ASRDataset(train_wav_paths, train_textgrid_paths, Sy_phoneme, Sy_word, config)
	valid_dataset = ASRDataset(valid_wav_paths, valid_textgrid_paths, Sy_phoneme, Sy_word, config)
	test_dataset = ASRDataset(test_wav_paths, test_textgrid_paths, Sy_phoneme, Sy_word, config)

	return train_dataset, valid_dataset, test_dataset

class ASRDataset(torch.utils.data.Dataset):
	def __init__(self, wav_paths, textgrid_paths, Sy_phoneme, Sy_word, config):
		"""
		wav_paths: list of strings (wav file paths)
		textgrid_paths: list of strings (textgrid for each wav file)
		Sy_phoneme: list of strings (all possible phonemes)
		Sy_word: list of strings (all possible words)
		config: Config object (contains info about model and training)
		"""
		self.wav_paths = wav_paths # list of wav file paths
		self.textgrid_paths = textgrid_paths # list of textgrid file paths
		self.length_mean = config.pretraining_length_mean
		self.length_var = config.pretraining_length_var
		self.Sy_phoneme = Sy_phoneme
		self.Sy_word = Sy_word
		self.phone_downsample_factor = config.phone_downsample_factor
		self.word_downsample_factor = config.word_downsample_factor
		
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.pretraining_batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsASR())

	def __len__(self):
		return len(self.wav_paths)

	def __getitem__(self, idx):
		x, fs = sf.read(self.wav_paths[idx])

		tg = textgrid.TextGrid()
		tg.read(self.textgrid_paths[idx])

		y_phoneme = []
		for phoneme in tg.getList("phones")[0]:
			duration = phoneme.maxTime - phoneme.minTime
			phoneme_index = self.Sy_phoneme.index(phoneme.mark.rstrip("0123456789")) if phoneme.mark.rstrip("0123456789") in self.Sy_phoneme else -1
			if phoneme.mark == '': phoneme_index = -1
			y_phoneme += [phoneme_index] * round(duration * fs)

		y_word = []
		for word in tg.getList("words")[0]:
			duration = word.maxTime - word.minTime
			word_index = self.Sy_word.index(word.mark) if word.mark in self.Sy_word else -1
			# if word.mark == '': word_index = -1
			y_word += [word_index] * round(duration * fs)

		# Cut a snippet of length random_length from the audio
		random_length = round(fs * max(self.length_mean + self.length_var * torch.randn(1).item(), 0.5))
		if len(x) <= random_length:
			start = 0
		else:
			start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
		end = start + random_length

		x = x[start:end]
		y_phoneme = y_phoneme[start:end:self.phone_downsample_factor]
		y_word = y_word[start:end:self.word_downsample_factor]

		return (x, y_phoneme, y_word)

class CollateWavsASR:
	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, phoneme labels, word labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y_phoneme = []; y_word = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_phoneme_, y_word_ = batch[index]

			x.append(torch.tensor(x_).float())
			y_phoneme.append(torch.tensor(y_phoneme_).long())
			y_word.append(torch.tensor(y_word_).long())

		# pad all sequences to have same length
		T = max([len(x_) for x_ in x])
		U_phoneme = max([len(y_phoneme_) for y_phoneme_ in y_phoneme])
		U_word = max([len(y_word_) for y_word_ in y_word])
		for index in range(batch_size):
			x_pad_length = (T - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			y_pad_length = (U_phoneme - len(y_phoneme[index]))
			y_phoneme[index] = torch.nn.functional.pad(y_phoneme[index], (0,y_pad_length), value=-1)
			
			y_pad_length = (U_word - len(y_word[index]))
			y_word[index] = torch.nn.functional.pad(y_word[index], (0,y_pad_length), value=-1)

		x = torch.stack(x)
		y_phoneme = torch.stack(y_phoneme)
		y_word = torch.stack(y_word)

		return (x,y_phoneme, y_word)
