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
\\
To add a new csv, the necessary functions are in ```local/data_csv```.\\
If you want access to the data, please copy the contents of the folder ```/home/tthebau1/EDART/SpeechLLM/data/*```
\\
Currently under work for connectors analysis

SpeechLLM is a multi-modal Language Model (LLM) specifically trained to analyze and predict metadata from a speaker's turn in a conversation. This advanced model integrates a speech encoder to transform speech signals into meaningful speech representations. These embeddings, combined with text instructions, are then processed by the LLM to generate predictions.

The model inputs an speech audio file of **16 KHz** and predicts the following:
1. **SpeechActivity** : if the audio signal contains speech (True/False)
2. **Transcript** : ASR transcript of the audio
3. **Gender** of the speaker (Female/Male)
4. **Age** of the speaker (number)
5. **Accent** of the speaker (Africa/America/Celtic/Europe/Oceania/South-Asia/South-East-Asia)
6. **Emotion** of the speaker (Happy/Sad/Anger/Neutral/Frustrated)
