import numpy as np
import torch
from tqdm import tqdm # for displaying progress bar
import os
from data import SLUDataset, ASRDataset
from models import PretrainedModel, Model
import pandas as pd

class Trainer:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		if isinstance(self.model, PretrainedModel):
			self.lr = config.pretraining_lr
			self.checkpoint_path = os.path.join(self.config.folder, "pretraining")
		else:
			self.lr = config.training_lr
			self.checkpoint_path = os.path.join(self.config.folder, "training")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.epoch = 0
		self.df = None

	def load_checkpoint(self):
		if os.path.isfile(os.path.join(self.checkpoint_path, "model_state.pth")):
			try:
				if self.model.is_cuda:
					self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth")))
				else:
					self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth"), map_location="cpu"))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def save_checkpoint(self):
		try:
			torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
		except:
			print("Could not save model")

	def log(self, results):
		if self.df is None:
			self.df = pd.DataFrame(columns=[field for field in results])
		self.df.loc[len(self.df)] = results
		self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

	def train(self, dataset, print_interval=100):
		# TODO: refactor to remove if-statement?
		if isinstance(dataset, ASRDataset):
			train_phone_acc = 0
			train_phone_loss = 0
			train_word_acc = 0
			train_word_loss = 0
			num_examples = 0
			self.model.train()
			for idx, batch in enumerate(tqdm(dataset.loader)):
				x,y_phoneme,y_word = batch
				batch_size = len(x)
				num_examples += batch_size
				phoneme_loss, word_loss, phoneme_acc, word_acc = self.model(x,y_phoneme,y_word)
				if self.config.pretraining_type == 1: loss = phoneme_loss
				if self.config.pretraining_type == 2: loss = phoneme_loss + word_loss
				if self.config.pretraining_type == 3: loss = word_loss
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				train_phone_loss += phoneme_loss.cpu().data.numpy().item() * batch_size
				train_word_loss += word_loss.cpu().data.numpy().item() * batch_size
				train_phone_acc += phoneme_acc.cpu().data.numpy().item() * batch_size
				train_word_acc += word_acc.cpu().data.numpy().item() * batch_size
				if idx % print_interval == 0:
					print("phoneme loss: " + str(phoneme_loss.cpu().data.numpy().item()))
					print("word loss: " + str(word_loss.cpu().data.numpy().item()))
					print("phoneme acc: " + str(phoneme_acc.cpu().data.numpy().item()))
					print("word acc: " + str(word_acc.cpu().data.numpy().item()))
			train_phone_loss /= num_examples
			train_phone_acc /= num_examples
			train_word_loss /= num_examples
			train_word_acc /= num_examples
			results = {"phone_loss" : train_phone_loss, "phone_acc" : train_phone_acc, "word_loss" : train_word_loss, "word_acc" : train_word_acc, "set": "train"}
			self.log(results)
			self.epoch += 1
			return train_phone_acc, train_phone_loss, train_word_acc, train_word_loss
		else: # SLUDataset
			train_intent_acc = 0
			train_intent_loss = 0
			num_examples = 0
			self.model.train()
			self.model.print_frozen()
			for idx, batch in enumerate(tqdm(dataset.loader)):
				x,y_intent = batch
				batch_size = len(x)
				num_examples += batch_size
				intent_loss, intent_acc = self.model(x,y_intent)
				loss = intent_loss
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				train_intent_loss += intent_loss.cpu().data.numpy().item() * batch_size
				train_intent_acc += intent_acc.cpu().data.numpy().item() * batch_size
				if idx % print_interval == 0:
					print("intent loss: " + str(intent_loss.cpu().data.numpy().item()))
					print("intent acc: " + str(intent_acc.cpu().data.numpy().item()))
					if self.model.seq2seq:
						self.model.cpu(); self.model.is_cuda = False
						x = x.cpu(); y_intent = y_intent.cpu()
						print("seq2seq output")
						self.model.eval()
						print("guess: " + self.model.decode_intents(x)[0])
						print("truth: " + self.model.one_hot_to_string(y_intent[0],self.model.Sy_intent))
						self.model.train()
						self.model.cuda(); self.model.is_cuda = True
			train_intent_loss /= num_examples
			train_intent_acc /= num_examples
			self.model.unfreeze_one_layer()
			results = {"intent_loss" : train_intent_loss, "intent_acc" : train_intent_acc, "set": "train"}
			self.log(results)
			self.epoch += 1
			return train_intent_acc, train_intent_loss

	def test(self, dataset):
		if isinstance(dataset, ASRDataset):
			test_phone_acc = 0
			test_phone_loss = 0
			test_word_acc = 0
			test_word_loss = 0
			num_examples = 0
			self.model.eval()
			for idx, batch in enumerate(dataset.loader):
				x,y_phoneme,y_word = batch
				batch_size = len(x)
				num_examples += batch_size
				phoneme_loss, word_loss, phoneme_acc, word_acc = self.model(x,y_phoneme,y_word)
				test_phone_loss += phoneme_loss.cpu().data.numpy().item() * batch_size
				test_word_loss += word_loss.cpu().data.numpy().item() * batch_size
				test_phone_acc += phoneme_acc.cpu().data.numpy().item() * batch_size
				test_word_acc += word_acc.cpu().data.numpy().item() * batch_size
			test_phone_loss /= num_examples
			test_phone_acc /= num_examples
			test_word_loss /= num_examples
			test_word_acc /= num_examples
			results = {"phone_loss" : test_phone_loss, "phone_acc" : test_phone_acc, "word_loss" : test_word_loss, "word_acc" : test_word_acc,"set": "valid"}
			self.log(results)
			return test_phone_acc, test_phone_loss, test_word_acc, test_word_loss 
		else:
			test_intent_acc = 0
			test_intent_loss = 0
			num_examples = 0
			self.model.eval()
			self.model.cpu(); self.model.is_cuda = False # beam search is memory-intensive; do on CPU for now
			for idx, batch in enumerate(dataset.loader):
				x,y_intent = batch
				batch_size = len(x)
				num_examples += batch_size
				intent_loss, intent_acc = self.model(x,y_intent)
				test_intent_loss += intent_loss.cpu().data.numpy().item() * batch_size
				test_intent_acc += intent_acc.cpu().data.numpy().item() * batch_size
				if self.model.seq2seq and self.epoch > 1:
					print("decoding batch %d" % idx)
					guess_strings = np.array(self.model.decode_intents(x))
					truth_strings = np.array([self.model.one_hot_to_string(y_intent[i],self.model.Sy_intent) for i in range(batch_size)])
					test_intent_acc += (guess_strings == truth_strings).mean() * batch_size
					print("acc: " + str((guess_strings == truth_strings).mean()))
					print("guess: " + guess_strings[0])
					print("truth: " + truth_strings[0])
			self.model.cuda(); self.model.is_cuda = True
			test_intent_loss /= num_examples
			test_intent_acc /= num_examples
			results = {"intent_loss" : test_intent_loss, "intent_acc" : test_intent_acc, "set": "valid"}
			self.log(results)
			return test_intent_acc, test_intent_loss 
