import torch
import numpy as np
from models import PretrainedModel, Model
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='run ASR pre-training')
parser.add_argument('--train', action='store_true', help='run SLU training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
pretrain = args.pretrain
train = args.train
restart = args.restart
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

if pretrain:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

	# Initialize base model
	pretrained_model = PretrainedModel(config=config)

	# Train the base model
	trainer = Trainer(model=pretrained_model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.pretraining_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		train_phone_acc, train_phone_loss, train_word_acc, train_word_loss = trainer.train(train_dataset)
		valid_phone_acc, valid_phone_loss, valid_word_acc, valid_word_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
		print("*phonemes*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_phone_acc, train_phone_loss, valid_phone_acc, valid_phone_loss) )
		print("*words*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_word_acc, train_word_loss, valid_word_acc, valid_word_loss) )

		trainer.save_checkpoint()

if train:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

	# Initialize final model
	model = Model(config=config)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.train(train_dataset)
		valid_intent_acc, valid_intent_loss = trainer.test(valid_dataset)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )

		trainer.save_checkpoint()

	test_intent_acc, test_intent_loss = trainer.test(test_dataset)
	print("========= Test results =========")
	print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )
