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
		
	def train(self, dataset, print_interval=100, randomize_length=True):
		self.epoch += 1
		if self.epoch < len(self.config.pretraining_length_schedule): 
			dataset.max_length = int(self.config.pretraining_length_schedule[self.epoch] * self.config.fs)
		else:
			dataset.max_length = int(self.config.pretraining_length_schedule[-1] * self.config.fs)

		train_phone_acc = 0
		train_phone_loss = 0
		train_word_acc = 0
		train_word_loss = 0
		num_examples = 0
		self.model.train()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			if randomize_length: dataset.max_length = int((torch.rand(1)*6 + 0.5) * self.config.fs)
			x,y_phoneme,y_word = batch
			batch_size = len(x)
			print(x.shape)
			num_examples += batch_size
			phoneme_loss, word_loss, phoneme_acc, word_acc = self.model(x,y_phoneme,y_word)
			loss = phoneme_loss + word_loss
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
		# train_acc = train_acc
		return train_phone_acc, train_phone_loss, train_word_acc, train_word_loss

	def test(self, dataset):
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
			# acc = (y * y_hat.cpu()).sum(1).mean()
			test_phone_loss += phoneme_loss.cpu().data.numpy().item() * batch_size
			test_word_loss += word_loss.cpu().data.numpy().item() * batch_size
			test_phone_acc += phoneme_acc.cpu().data.numpy().item() * batch_size
			test_word_acc += word_acc.cpu().data.numpy().item() * batch_size
			# test_acc += acc * batch_size
		test_phone_loss /= num_examples
		test_phone_acc /= num_examples
		test_word_loss /= num_examples
		test_word_acc /= num_examples
		return test_phone_acc, test_phone_loss, test_word_acc, test_word_loss 