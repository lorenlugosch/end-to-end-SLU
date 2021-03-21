# End-to-End Spoken Language Understanding (SLU) in PyTorch
This repo contains PyTorch code for training end-to-end SLU models used in the papers "[Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670)" and "[Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models](https://arxiv.org/abs/1910.09463)".

If you have any questions about this code or have problems getting it to work, please send me an email at ```<the email address listed for Loren in the paper>```.

(**Note:** See the SpeechBrain repository for [a simpler recipe for Fluent Speech Commands](https://github.com/speechbrain/speechbrain/tree/develop/recipes/fluent-speech-commands) and other SLU benchmarks.)

## Dependencies
PyTorch, torchaudio, numpy, soundfile, pandas, tqdm, textgrid.py

## Training
First, change the ```asr_path``` and/or ```slu_path``` in the config file (like ```experiments/no_unfreezing.cfg```, or whichever experiment you want to run) to point to where the LibriSpeech data and/or Fluent Speech Commands data are stored on your computer.

_SLU training:_ To train the model on an SLU dataset, run the following command:
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
If you find this repo or our Fluent Speech Commands dataset useful, please cite our papers:

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding", Interspeech 2019.
- Loren Lugosch, Brett Meyer, Derek Nowrouzezahrai, and Mirco Ravanelli, "Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models", ICASSP 2020.
