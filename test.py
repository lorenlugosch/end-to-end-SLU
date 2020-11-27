import torch
import numpy as np
from models import PretrainedModel, Model
from data import get_ASR_datasets, get_SLU_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, required=True, help='path to config file with hyperparameters, etc.')
parser.add_argument('--error_path', type=str, required=True, help='path to store list of files with predicted errors.')
args = parser.parse_args()
restart = args.restart
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

# Initialize final model
model = Model(config=config)

# Train the final model
trainer = Trainer(model=model, config=config)
if restart: trainer.load_checkpoint()


test_intent_acc, test_intent_loss = trainer.get_error(test_dataset, error_path=args.error_path)
print("========= Test results =========")
print("*intents*| test accuracy: %.2f| test loss: %.2f\n" % (test_intent_acc, test_intent_loss) )
