# Speech Model Pre-training for End-to-End Spoken Language Understanding
This repo contains the code for the paper "[Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670)".
Our paper introduces the [Fluent Speech Commands](http://www.fluent.ai/research/fluent-speech-commands/) dataset and explores useful pre-training strategies for end-to-end spoken language understanding.

I'm going to keep messing around with this code; if you want a version that is guaranteed to give you the results in the paper, use [this commit](https://github.com/lorenlugosch/pretrain_speech_model/tree/2213a122718cbbc9bd8b6762f9c87c9cef0b04d3).

If you have any questions about this code or have problems getting it to work, please send me an email at ```<the email address listed for Loren in the paper>```.

## Dependencies
PyTorch, numpy, soundfile, pandas, tqdm, textgrid.py

## Training
First, change the ```asr_path``` and/or ```slu_path``` in the config file (like ```experiments/no_unfreezing.cfg```, or whichever experiment you want to run) to point to where the LibriSpeech data and/or Fluent Speech Commands data are stored on your computer.

_SLU training:_ To train the model on Fluent Speech Commands, run the following command:
```
python main.py --train --config_path=<path to .cfg>
```

_ASR pre-training:_ **Note:** the experiment folders in this repo already have a pre-trained LibriSpeech model that you can use. LibriSpeech is pretty big (>100 GB uncompressed), so don't do this part unless you want to re-run the pre-training part with different hyperparameters. If you want to do this, you will first need to download our LibriSpeech alignments [here](https://zenodo.org/record/2619474#.XKDP2VNKg1g), put them in a folder called "text", and put the LibriSpeech audio in a folder called "audio". To pre-train the model on LibriSpeech, run the following command:
```
python main.py --pretrain --config_path=<path to .cfg>
```

## Inference
You can perform inference with a trained SLU model as follows (thanks, Nathan Folkman!):
```python
import data
import models
import soundfile as sf
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

signal, _ = sf.read("test.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0)

model.decode_intents(signal)
```
The ```test.wav``` file included with this repo has a recording of me saying "Hey computer, could you turn the lights on in the kitchen please?", and so the inferred intent should be ```{"activate", "lights", "kitchen"}```.

## Citation
If you find this repo or our Fluent Speech Commands dataset useful, please cite our paper:

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", arXiv:1904.03670, 2019.
