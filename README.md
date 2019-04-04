# Speech Model Pre-training for End-to-End Spoken Language Understanding
This repo contains the code for the paper "Speech Model Pre-training for End-to-End Spoken Language Understanding".

If you have any questions about this code or have problems getting it to work, please send me an email at ```<the email address listed for me in the paper>```.

## Dependencies
PyTorch, numpy, soundfile, pandas, tqdm, textgrid.py

## Usage
First, change the ```asr_path``` and/or ```slu_path``` in the config file (like ```experiments/no_unfreezing.cfg```, or whichever experiment you want to run) to point to where the LibriSpeech data and/or Fluent Speech Commands data are stored on your computer.

_ASR pre-training:_ Note that the experiment folders in this repo already have a pre-trained LibriSpeech model that you can use. LibriSpeech is pretty big (>100 GB uncompressed), so don't do this part unless you want to re-run the pre-training part with different hyperparameters. To pre-train the model on LibriSpeech, run the following command:
```
python main.py --pretrain --config_path=<path to .cfg>
```

_SLU training:_ To train the model on Fluent Speech Commands, run the following command:
```
python main.py --train --config_path=<path to .cfg>
```

## Citation
If you find this repo or our Fluent Speech Commands dataset useful, please cite our paper:

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", arXiv, 2019.
