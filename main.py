import torch
import numpy as np
from models import PretrainedModel
from data import get_datasets, read_config
from training import Trainer

# Read config
config = read_config("cfg/pretrained_model.cfg")
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Generate datasets from folder
path = "/home/ubuntu/data/librispeech"
train_dataset, valid_dataset, test_dataset = get_datasets(path, config)

# Initialize model
model = PretrainedModel(config=config)
# model = torch.load("pretrained_model_no_retraining.pth")

# Train the model
trainer = Trainer(model=model, config=config)
# trainer.load_checkpoint(checkpoint_path)

for epoch in range(config.num_epochs):
	print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
	train_phone_acc, train_phone_loss, train_word_acc, train_word_loss = trainer.train(train_dataset)
	valid_phone_acc, valid_phone_loss, valid_word_acc, valid_word_loss = trainer.test(valid_dataset)

	print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
	print("*phonemes*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_phone_acc, train_phone_loss, valid_phone_acc, valid_phone_loss) )
	print("*words*| train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_word_acc, train_word_loss, valid_word_acc, valid_word_loss) )

	torch.save_checkpoint(model, "pretrained_model_state.pth")
