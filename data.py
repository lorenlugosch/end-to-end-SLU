import torch
import torch.utils.data
import os, glob
from collections import Counter
import soundfile as sf
import python_speech_features
import numpy as np
import configparser

class Config:
	def __init__(self):
		self.use_sincnet = True

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[phoneme_module]
	config.use_sincnet=(parser.get("phoneme_module", "use_sincnet") == "True")
	config.fs=int(parser.get("phoneme_module", "fs"))

	config.cnn_N_filt=[int(x) for x in parser.get("phoneme_module", "cnn_N_filt").split(",")]
	config.cnn_len_filt=[int(x) for x in parser.get("phoneme_module", "cnn_len_filt").split(",")]
	config.cnn_stride=[int(x) for x in parser.get("phoneme_module", "cnn_stride").split(",")]
	config.cnn_max_pool_len=[int(x) for x in parser.get("phoneme_module", "cnn_max_pool_len").split(",")]
	config.cnn_use_laynorm_inp=(parser.get("phoneme_module", "cnn_use_laynorm_inp") == "True")
	config.cnn_use_batchnorm_inp=(parser.get("phoneme_module", "cnn_use_batchnorm_inp") == "True")
	config.cnn_use_laynorm=[(x == "True") for x in parser.get("phoneme_module", "cnn_use_laynorm").split(",")]
	config.cnn_use_batchnorm=[(x == "True") for x in parser.get("phoneme_module", "cnn_use_batchnorm").split(",")]
	config.cnn_act=[x for x in parser.get("phoneme_module", "cnn_act").split(",")]
	config.cnn_drop=[float(x) for x in parser.get("phoneme_module", "cnn_drop").split(",")]

	config.phone_rnn_type=parser.get("phoneme_module", "phone_rnn_type")
	config.phone_rnn_lay=[int(x) for x in parser.get("phoneme_module", "phone_rnn_lay").split(",")]
	config.phone_downsample_len=[int(x) for x in parser.get("phoneme_module", "phone_downsample_len").split(",")]
	config.phone_downsample_type=[x for x in parser.get("phoneme_module", "phone_downsample_type").split(",")]
	config.phone_rnn_drop=[float(x) for x in parser.get("phoneme_module", "phone_rnn_drop").split(",")]
	config.phone_rnn_bidirectional=(parser.get("phoneme_module", "phone_rnn_bidirectional") == True)

	#[word_module]
	config.word_rnn_type=parser.get("word_module", "word_rnn_type")
	config.word_rnn_lay=[int(x) for x in parser.get("word_module", "word_rnn_lay").split(",")]
	config.word_downsample_len=[int(x) for x in parser.get("word_module", "word_downsample_len").split(",")]
	config.word_downsample_type=[x for x in parser.get("word_module", "word_downsample_type").split(",")]
	config.word_rnn_drop=[float(x) for x in parser.get("word_module", "word_rnn_drop").split(",")]
	config.word_rnn_bidirectional=(parser.get("word_module", "word_rnn_bidirectional") == True)
	config.vocabulary_size=int(parser.get("word_module", "vocabulary_size"))

	#[pretraining]
	pretraining_lr=0.001
	pretraining_batch_size=64
	pretraining_num_epochs=20
	pretraining_seed=1234
	pretraining_length_schedule=0.5,1.0,2.0,4.0,8.0
	config.lr=float(parser.get("pretraining", "lr"))
	config.batch_size=int(parser.get("pretraining", "batch_size"))
	config.num_epochs=int(parser.get("pretraining", "num_epochs"))
	config.seed=int(parser.get("pretraining", "seed"))
	config.pretraining_length_schedule=int(parser.get("pretraining", "seed"))

	# compute downsample factor (divide T by this number)
	config.downsample_factor = 1
	for factor in config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len + config.word_downsample_len:
		config.downsample_factor *= factor

	return config

def get_datasets(base_path, config):
	"""
	base_path: string (directory containing the speech dataset)
		e.g., "/home/data/librispeech"

		Assumes this directory contains the following two directories:
			"LibriSpeech" : wav files (split into )
			"aligned" : alignments for each wav

	config: Config object (contains info about model and training)
	"""

	# Get only files with a label
	train_textgrid_paths = glob.glob(path_to_librispeech + "/aligned/train*/*/*/*.TextGrid")
	train_wav_paths = [path.replace("aligned", "LibriSpeech").replace(".TextGrid", ".wav") for path in train_textgrid_paths]
	valid_textgrid_paths = glob.glob(path_to_librispeech + "/aligned/dev*/*/*/*.TextGrid")
	valid_wav_paths = [path.replace("aligned", "LibriSpeech").replace(".TextGrid", ".wav") for path in valid_textgrid_paths]
	test_textgrid_paths = glob.glob(path_to_librispeech + "/aligned/test*/*/*/*.TextGrid")
	test_wav_paths = [path.replace("aligned", "LibriSpeech").replace(".TextGrid", ".wav") for path in test_textgrid_paths]
	
	# Get list of phonemes and words
	print("Getting vocabulary...")
	phoneme_counter = Counter()
	word_counter = Counter()
	for path in valid_textgrid_paths:
		tg = textgrid.TextGrid()
		tg.read(path)
		phoneme_counter.update([phone.mark.rstrip("0123456789") for phone in tg.getList("phones")[0] if phone.mark != ''])
		word_counter.update([word.mark for word in tg.getList("words")[0] if word.mark != ''])

	Sy_phoneme = list(phoneme_counter)
	Sy_word = [w[0] for w in word_counter.most_common(config.vocabulary_size)]
	config.num_phonemes = len(Sy_phoneme)

	# Create dataset objects
	train_dataset = SpeechCommandDataset(train_wav_paths, train_textgrid_paths, Sy_phoneme, Sy_word, config)
	valid_dataset = SpeechCommandDataset(valid_wav_paths, valid_textgrid_paths, Sy_phoneme, Sy_word, config)
	test_dataset = SpeechCommandDataset(test_wav_paths, test_textgrid_paths, Sy_phoneme, Sy_word, config)

	return train_dataset, valid_dataset, test_dataset

class SpeechCommandDataset(torch.utils.data.Dataset):
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
		self.max_length = 80000 # truncate audios longer than this
		self.Sy_phoneme = Sy_phoneme
		self.Sy_word = Sy_word
		self.downsample_factor = config.downsample_factor
		
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.batch_size, num_workers=4, shuffle=True, collate_fn=CollateWavs())

	def __len__(self):
		return len(self.wav_paths)

	def __getitem__(self, idx):
		x, fs = sf.read(self.wav_paths[idx])

		# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
		# if config.use_fbank:
		# eps = 1e-8
		# fbank = python_speech_features.fbank(x, nfilt=40, winfunc=np.hamming)
		# fbank = np.concatenate([fbank[1].reshape(-1,1), fbank[0]], axis=1) + eps
		# fbank = np.log(fbank)
		# fbank = (fbank - fbank.mean(0))
		# fbank = fbank/(np.sqrt(fbank.var(0)))

		tg = textgrid.TextGrid()
		tg.read(self.textgrid_paths[idx])

		y_phoneme = []
		for phoneme in tg.getList("phones")[0]:
			duration = phoneme.maxTime - phoneme.minTime
			phoneme_index = Sy_phoneme.index(phoneme.mark.rstrip("0123456789")) if phoneme.mark.rstrip("0123456789") in Sy_phoneme else -1
			if phoneme.mark == '': phoneme_index = -1
			y_phoneme += [phoneme_index] * round(duration * fs)

		y_word = []
		for word in tg.getList("words")[0]:
			duration = word.maxTime - word.minTime
			word_index = Sy_word.index(word.mark) if word.mark in Sy_word else -1
			if word.mark == '': word_index = -1
			y_word += [word_index] * round(duration * fs)

		if len(x) < self.max_length:
			start = 0
		else:
			start = torch.randint(low=0, high=len(x)-self.max_length, size=(1,)).item()
		end = start + self.max_length

		x = x[start:end]
		y_phoneme = y_phoneme[start:end:self.downsample_factor]
		y_word = y_word[start:end:self.downsample_factor]

		return (x, y_phoneme, y_word)

def one_hot(labels, S):
	"""
	labels : LongTensor of shape (batch size,)
	S : integer

	Convert batch of integer labels to one-hot vectors of dimension S (# of possible letters).
	"""

	out = torch.zeros(labels.shape[0], S)
	for i in range(0, labels.shape[0]):
		out[i, labels[i]] = 1
	return out

class CollateWavs:
	# def __init__(self, Sy):
	# 	self.Sy = Sy

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
		U = max([len(y_phoneme_) for y_phoneme_ in y_phoneme])
		for index in range(batch_size):
			x_pad_length = (T - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			y_pad_length = (U - len(y_phoneme[index]))
			y_phoneme[index] = torch.nn.functional.pad(y_phoneme[index], (0,y_pad_length))
			y_word[index] = torch.nn.functional.pad(y_word[index], (0,y_pad_length))

		x = torch.stack(x)
		y_phoneme = torch.stack(y_phoneme)
		y_word = torch.stack(y_word)
		# y = one_hot(y, len(self.Sy))

		return (x,y_phoneme, y_word)
