import torch
import numpy as np
from models import PretrainedModel
from data import get_datasets, read_config
from training import Trainer

# Read config
config = read_config("cfg/pretrained_model.cfg")
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Generate datasets from folder
path = "/scratch/lugosch/librispeech"
train_dataset, valid_dataset, test_dataset = get_datasets(path, config)

# Initialize model
model = PretrainedModel(config=config)

# Train the model
trainer = Trainer(model=model, config=config)
for epoch in range(config.num_epochs):
	print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
	train_acc, train_loss = trainer.train(train_dataset)
	valid_acc, valid_loss = trainer.test(valid_dataset)

	print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
	print("train accuracy: %.2f| train loss: %.2f| valid accuracy: %.2f| valid loss: %.2f\n" % (train_acc, train_loss, valid_acc, valid_loss) )

	torch.save(model, "pretrained_model_state.pth")
