import torch
from tqdm import tqdm # for displaying progress bar

class Trainer:
	def __init__(self, model, config):
		self.model = model
		if torch.cuda.is_available(): self.model = self.model.cuda()
		self.lr = config.lr
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.config = config
		self.epoch = -1
		
	def train(self, dataset, print_interval=100):
		self.epoch += 1
		if self.epoch < len(self.config.pretraining_length_schedule): 
			dataset.max_length = int(self.config.pretraining_length_schedule[self.epoch] * self.config.fs)
		else:
			dataset.max_length = int(self.config.pretraining_length_schedule[-1] * self.config.fs)

		train_acc = 0
		train_loss = 0
		num_examples = 0
		self.model.train()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,y_phoneme,y_word = batch
			batch_size = len(x)
			num_examples += batch_size
			phoneme_loss, word_loss = self.model(x,y_phoneme,y_word)
			loss = phoneme_loss + word_loss
			# acc = (y * y_hat.cpu()).sum(1).mean()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			train_loss += loss.cpu().data.numpy().item() * batch_size
			# train_acc += acc * batch_size
			if idx % print_interval == 0:
				print("phoneme loss: " + str(phoneme_loss.cpu().data.numpy().item()))
				print("word loss: " + str(word_loss.cpu().data.numpy().item()))
		train_loss /= num_examples
		train_acc /= num_examples
		train_acc = train_acc
		return train_acc, train_loss

	def test(self, dataset):
		test_acc = 0
		test_loss = 0
		num_examples = 0
		self.model.eval()
		for idx, batch in enumerate(dataset.loader):
			x,y_phoneme,y_word = batch
			batch_size = len(x)
			num_examples += batch_size
			phoneme_loss, word_loss = self.model(x,y_phoneme,y_word)
			loss = phoneme_loss + word_loss
			# acc = (y * y_hat.cpu()).sum(1).mean()
			test_loss += loss.cpu().data.numpy().item() * batch_size
			# test_acc += acc * batch_size
		test_loss /= num_examples
		test_acc /= num_examples
		test_acc = test_acc
		return test_acc, test_loss