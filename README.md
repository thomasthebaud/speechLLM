# SpeechLLM
Repository adapted from https://github.com/skit-ai/SpeechLLM/tree/main

Propose the training and testing of Speech LLMs for speaker characterization and summarization, adapted for the CLSP grid.

# Options Available right now
## Datasets
The following datasets have been adapted and can be used for training, with the following fields:
| dataset | train split | dev split | test split | fields |
| ------- | ------ | ------ | ------ | ------ |
| Crema-D | train | dev | test | 6 Cat. emotions, gender, transcript |
| Common Voice EN v11 |  train | dev | test | nationality, age (decade), gender, accent (16 nationalities), transcript |
| IEMOCAP | ses01-03 | ses04 | ses05 | 4 Cat. emotions, gender |
| Librispeech | train-clean-100, train-clean-360, train-other-500 | dev-clean, dev-other | test-clean, test-other | gender, transcript |
| MSP podcast | train | validation | test | 8 Cat. emotions, gender | 
| Switchboard | train | validation | test | transcript, summary |
| VoxCeleb1 | dev | test | test | gender, accent (from nationality) |
| VoxCeleb2-AE | dev | test | test | gender, age, accent (from nationality) |
| WSJ0 | si_tr_s | si_dt_05 | si_et_05 | gender, transcript |

Most datasets use the original splits.
Voxceleb datasets are using the same validation and test, as the models are trained to optimize validation summary loss anyway.
<br>
To add a new csv, the necessary functions are in ```local/data_csv```.
<br>
If you want access to the data, please copy the contents of the folder ```/home/tthebau1/EDART/SpeechLLM/data/*``` in your own ```data/``` folder.

## Architectures
In general, all parameters and their defaults values can be adjusted in the get_model_config() function in ```utils.py```.
The base system currently uses:
- A [WavLM base plus](https://huggingface.co/microsoft/wavlm-base-plus) feature encoder, with 768 dimensional output features. It can be replaced by any hugging face encoder, by modifying the parameters: 
    - ```--encoder 'microsoft/wavlm-base-plus'``` (currently accepts ```facebook/hubert-xlarge-ll60k```, ```microsoft/wavlm-large```, ```microsoft/wavlm-base-plus```, ```MFCC```, the list can be expanded in ```models/encoder.py```)
    - ```--encoder-dim 768``` to adjust the desired output dimension.

- A windowed meanpooling layer, with a ratio ```--meanpool 5```
- A CNN connector, which uses the following parameters:
    - ```--connector 'cnn'``` for the type of connector. more types and architectures can be added in the ```models/connector.py``` file.
    - ```--connector-k 2``` for the stride
    - ```--connector-layers 2``` for the number of layers in case of a MLP
    - ```--connector-dim 1024``` for the output dimension of features
- A LLM. Currently uses a [Tiny LLAMA](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), you can change it by adjusting the parameter:
    - ```--llm 'TinyLlama-1.1B-Chat-v1.0'```

## Parameters

### For Training
there is one learning rate for the LoRa adaptors of the LLM and the connector (```--lr 0.0001```), and one learning rate for the feature extractor (```--encoder-lr 0.000001```, by default 50 times lower than the base lr).
<br>
If ```--no-lora``` is passed, the LLM is frozen. <br>
If ```--ft-encoder``` is passed, the encoder is fine-tuned. <br>
If ```--use-text``` is passed, transcripts will be added as inputs when available, with a probability ```--prob-text``` during training. <br>
If ```--no-audio``` is passed, only the transcripts will be used, no encoder nor connector will be initialized nor used. <br>

### Configurations
Training configurations are defined in the folder ```configs```, they can be passed as arguments ```--use-config summarize_switchboard.json```.
They define which dataset should be used for training, testing and validation, and which tasks should be used for each.
<br>
Trains up to ```--total-training-epoch``` maximum training epochs. top 3 models saved in ```checkpoints/```. Uses ```--epoch-to-test``` to test a specific epoch.

### Naming
```--group``` is used by Wandb to put the experiment in a given group.<br>
```--nickname``` is used to differentiate experiments and models with similar architecture but variations in configurations.

# Installation and running an experiment
## Installation
- **Conda**: conda environment is available in ```environment.yml```, use 
```
conda env create -f environment.yml
``` 
- **Pip**: pip environment is available in ```requirements.txt```, use 
```
pip install -r requirements.txt
```

## Launching a training
To train a network, use 
```
sbatch launch/$expe_series/train/$your_script.sh
```
To test it, use 
```
sbatch launch/$expe_series/test/$your_script.sh
```
The experiments with simple linear layer for speaker characterization are available in ```launch/ASRU2025```, allowing partial reproduction of this article: [*Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM*](https://arxiv.org/pdf/2508.04795).
Please cite this if you use those experiments:
```
@article{thebaud2025enhancing,
  title={Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM},
  author={Thebaud, Thomas and Lu, Yen-Ju and Wiesner, Matthew and Viechnicki, Peter and Dehak, Najim},
  journal={arXiv preprint arXiv:2508.04795},
  year={2025}
}
```
<br>
The experiments with CNN connector for audio summarization are available in ```launch/ICASSP2025```. The article was not submitted, it will be to a different venue.
<br>

# Contact
If you have any question, please contact Thomas Thebaud on slack, or use tthebau1@jhu.edu.