import torch
import torch.utils.data
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
	config.dataset_subset_percentage=float(parser.get("training", "dataset_subset_percentage"))
	config.train_wording_path=parser.get("training", "train_wording_path")
	if config.train_wording_path=="None": config.train_wording_path = None
	config.test_wording_path=parser.get("training", "test_wording_path")
	if config.test_wording_path=="None": config.test_wording_path = None

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
	train_df = pd.read_csv(os.path.join(base_path, "data", "train_data.csv"))
	valid_df = pd.read_csv(os.path.join(base_path, "data", "valid_data.csv"))
	test_df = pd.read_csv(os.path.join(base_path, "data", "test_data.csv"))
	
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

	# Select random subset of training data
	if config.dataset_subset_percentage < 1:
		subset_size = round(config.dataset_subset_percentage * len(train_df))
		train_df = train_df.loc[np.random.choice(len(train_df), subset_size, replace=False)]
		train_df = train_df.set_index(np.arange(len(train_df)))

	# Create dataset objects
	train_dataset = SLUDataset(train_df, base_path, Sy_intent, config)
	valid_dataset = SLUDataset(valid_df, base_path, Sy_intent, config)
	test_dataset = SLUDataset(test_df, base_path, Sy_intent, config)

	return train_dataset, valid_dataset, test_dataset

class SLUDataset(torch.utils.data.Dataset):
	def __init__(self, df, base_path, Sy_intent, config):
		"""
		df:
		Sy_intent: Dictionary (transcript --> slot values)
		config: Config object (contains info about model and training)
		"""
		self.df = df
		self.base_path = base_path
		self.max_length = 200000 # truncate audios longer than this
		self.Sy_intent = Sy_intent
		
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.training_batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsSLU())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		wav_path = os.path.join(self.base_path, self.df.loc[idx].path)
		x, fs = sf.read(wav_path)

		if len(x) <= self.max_length:
			start = 0
		else:
			start = torch.randint(low=0, high=len(x)-self.max_length, size=(1,)).item()
		end = start + self.max_length

		x = x[start:end]
		y_intent = [] 
		for slot in ["action", "object", "location"]:
			value = self.df.loc[idx][slot]
			y_intent.append(self.Sy_intent[slot][value])

		return (x, y_intent)

class CollateWavsSLU:
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
		T = max([len(x_) for x_ in x])
		for index in range(batch_size):
			x_pad_length = (T - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

		x = torch.stack(x)
		y_intent = torch.stack(y_intent)

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
