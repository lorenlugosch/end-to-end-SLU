import torch
import numpy as np
from models import PretrainedModel, Model
from data import get_datasets, read_config
from training import Trainer

# pre-train, train, or both
pretrain = False
train = True

# Read config
config = read_config("cfg/pretrained_model.cfg")
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Generate datasets from folder
path = "/scratch/lugosch/librispeech"
train_dataset, valid_dataset, test_dataset = get_ASR_datasets(path, config)

# Initialize base model
pretrained_model = PretrainedModel(config=config)

# Train the base model
trainer = Trainer(model=pretrained_model, config=config)
checkpoint_path = "pretrained_model/"
trainer.load_checkpoint(checkpoint_path)

if pretrain:
	for epoch in range(config.pretraining_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_phone_acc, train_phone_loss, train_word_acc, train_word_loss = trainer.train(train_dataset)
		valid_phone_acc, valid_phone_loss, valid_word_acc, valid_word_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("*phonemes*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_phone_acc, train_phone_loss, valid_phone_acc, valid_phone_loss) )
		print("*words*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_word_acc, train_word_loss, valid_word_acc, valid_word_loss) )

		trainer.save_checkpoint(epoch, checkpoint_path)

# # test dis bad boy out
# import soundfile as sf
# import matplotlib.pyplot as plt
# x, fs = sf.read("/scratch/lugosch/speech_commands/bed/e98cb283_nohash_0.wav") #sf.read("../../data/google_speech_commands/bed/e98cb283_nohash_0.wav")
# x = torch.stack([torch.tensor(x)]).float()
# phoneme_logits, word_logits = pretrained_model.compute_posteriors(x)
# plt.imshow(torch.nn.functional.softmax(phoneme_logits, dim=2)[0].data.cpu().numpy()); plt.show()
# plt.imshow(torch.nn.functional.softmax(word_logits, dim=2)[0].data.cpu().numpy()); plt.show()
# for word in word_logits.max(dim=1)[0].topk(5)[1][0]: # print top 5 words
# 	print(train_dataset.Sy_word[word.item()])

if train:
	# Generate datasets from folder
	path = "/scratch/lugosch/fluent_commands_dataset/"
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(path, config)

	# Initialize model
	model = Model(config=config, pretrained_model=pretrained_model)

	trainer = Trainer(model=model, config=config)
	checkpoint_path = "model/"
	trainer.load_checkpoint(checkpoint_path)

	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_intent_acc, train_intent_loss = trainer.train(train_dataset)
		valid_intent_acc, valid_intent_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )

		trainer.save_checkpoint(epoch, checkpoint_path)
