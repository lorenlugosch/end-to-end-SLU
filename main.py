import torch
import numpy as np
import pandas as pd
from models import PretrainedModel, Model, obtain_glove_embeddings
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse
import os

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='run ASR pre-training')
parser.add_argument('--train', action='store_true', help='run SLU training')
parser.add_argument('--pipeline_train', action='store_true', help='run SLU training in pipeline manner')
parser.add_argument('--get_words', action='store_true', help='get words from SLU pipeline')
parser.add_argument('--save_words_path', default="/tmp/word_transcriptions.csv", help='path to save audio transcription CSV file')
parser.add_argument('--postprocess_words', action='store_true', help='postprocess words obtained from SLU pipeline')
parser.add_argument('--use_semantic_embeddings', action='store_true', help='use Glove embeddings')
parser.add_argument('--semantic_embeddings_path', type=str, help='path for Glove embeddings')
parser.add_argument('--finetune_embedding', action='store_true', help='tune SLU embeddings')
parser.add_argument('--random_split', action='store_true', help='randomly split dataset')
parser.add_argument('--disjoint_split', action='store_true', help='split dataset with disjoint utterances in train set and test set')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
parser.add_argument('--pipeline_gold_train', action='store_true', help='run SLU training in pipeline manner with gold set utterances')
args = parser.parse_args()
pretrain = args.pretrain
train = args.train
pipeline_train = args.pipeline_train
pipeline_gold_train = args.pipeline_gold_train
get_words = args.get_words
postprocess_words = args.postprocess_words
restart = args.restart
config_path = args.config_path
use_semantic_embeddings = args.use_semantic_embeddings
semantic_embeddings_path = args.semantic_embeddings_path
finetune_embedding = args.finetune_embedding
random_split = args.random_split
disjoint_split = args.disjoint_split

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

if get_words:
	# Generate datasets
	Sy_word = []
	with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
		for line in f.readlines():
			Sy_word.append(line.rstrip("\n"))
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

	# Initialize final model
	model = Model(config=config)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	predicted_words, audio_paths = trainer.get_word_SLU(test_dataset,Sy_word, postprocess_words)
	df=pd.DataFrame({'audio path': audio_paths, 'predicted_words': predicted_words})
	df.to_csv(args.save_words_path, index=False)

if pipeline_train:
	# Generate datasets
	Sy_word = []
	with open(os.path.join(config.folder, "pretraining", "words.txt"), "r") as f:
		for line in f.readlines():
			Sy_word.append(line.rstrip("\n"))
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

	if postprocess_words:
		log_file="log_pipeline_postprocess.csv"
	else:
		if finetune_embedding:
			log_file="log_pieline_finetune.csv"
		else:
			log_file="log_pipeline.csv"
	
	# Initialize final model
	model = Model(config=config,pipeline=True,finetune=finetune_embedding)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset, postprocess_words,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset,  postprocess_words,log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
		if postprocess_words:
			trainer.save_checkpoint(model_path="model_state_postprocess.pth")
		else:
			if finetune_embedding:
				trainer.save_checkpoint(model_path="model_state_pipeline_finetune.pth")
			else:
				trainer.save_checkpoint(model_path="model_state_pipeline.pth")

	test_intent_acc, test_intent_loss = trainer.pipeline_test_decoder(test_dataset, postprocess_words,log_file=log_file)
	print("========= Test results =========")
	print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )

if pipeline_gold_train:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,use_gold_utterances=True,random_split=random_split, disjoint_split=disjoint_split)

	# Initialize final model
	if use_semantic_embeddings:
		glove_embeddings=obtain_glove_embeddings(semantic_embeddings_path, train_dataset.Sy_word )
		model = Model(config=config,pipeline=True, use_semantic_embeddings = use_semantic_embeddings, glove_embeddings=glove_embeddings)
	else:
		model = Model(config=config,pipeline=True, use_semantic_embeddings = False)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	log_file="log_pipeline_gold"
	only_model_path="only_gold_model_state"
	with_model_path="with_gold_model_state"

	if postprocess_words:
		log_file=log_file+"_postprocess"
		only_model_path=only_model_path + "_postprocess"
		with_model_path=with_model_path + "_postprocess"
	if disjoint_split:
		log_file=log_file+"_disjoint"
		only_model_path=only_model_path + "_disjoint"
		with_model_path=with_model_path + "_disjoint"
	elif random_split:
		log_file=log_file+"_random"
		only_model_path=only_model_path + "_random"
		with_model_path=with_model_path + "_random"

	if use_semantic_embeddings:
		log_file=log_file+"_glove"
		only_model_path=only_model_path + "_glove"
		with_model_path=with_model_path + "_glove"

	log_file=log_file+".csv"
	only_model_path=only_model_path + ".pth"
	with_model_path=with_model_path + ".pth"

	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset,gold=True,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset, postprocess_words,log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
		trainer.save_checkpoint(model_path=only_model_path)
	
	train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config,random_split=random_split, disjoint_split=disjoint_split)
	for epoch in range(config.training_num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		train_intent_acc, train_intent_loss = trainer.pipeline_train_decoder(train_dataset, postprocess_words,log_file=log_file)
		valid_intent_acc, valid_intent_loss = trainer.pipeline_test_decoder(valid_dataset, postprocess_words, log_file=log_file)

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.training_num_epochs))
		print("*intents*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_intent_acc, train_intent_loss, valid_intent_acc, valid_intent_loss) )
		trainer.save_checkpoint(model_path=with_model_path)

	test_intent_acc, test_intent_loss = trainer.pipeline_test_decoder(test_dataset, postprocess_words, log_file=log_file)
	print("========= Test results =========")
	print("*intents*| test accuracy: %.2f| test loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (test_intent_acc, test_intent_loss, valid_intent_acc, valid_intent_loss) )
