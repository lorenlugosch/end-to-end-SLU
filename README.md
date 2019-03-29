# pretrain_speech_model
This repo contains the code for replicating the experiments in the paper "Speech Model Pre-training for End-to-End Spoken Language Understanding".

## Usage
First, change the directories in the config file ```exp1.cfg``` to wherever the LibriSpeech data and Fluent Speech Commands data are stored on your computed. (We include a pre-trained model for LibriSpeech, so you don't have to run this.)

To pre-train the model on LibriSpeech, run the following command:
```
python main.py --pretrain --config_path=exp1.cfg
```

To train the model on Fluent Speech Commands, run the following command:
```
python main.py --train --config_path=exp1.cfg
```

## Citation
If you find this repo or our Fluent Speech Commands dataset useful, please cite our paper:

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", arXiv, 2019.
