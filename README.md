# SpeechLLM
Repository adapted from [(https://github.com/skit-ai/SpeechLLM/tree/main)]

Currently under work for connectors analysis

SpeechLLM is a multi-modal Language Model (LLM) specifically trained to analyze and predict metadata from a speaker's turn in a conversation. This advanced model integrates a speech encoder to transform speech signals into meaningful speech representations. These embeddings, combined with text instructions, are then processed by the LLM to generate predictions.

The model inputs an speech audio file of **16 KHz** and predicts the following:
1. **SpeechActivity** : if the audio signal contains speech (True/False)
2. **Transcript** : ASR transcript of the audio
3. **Gender** of the speaker (Female/Male)
4. **Age** of the speaker (Young/Middle-Age/Senior)
5. **Accent** of the speaker (Africa/America/Celtic/Europe/Oceania/South-Asia/South-East-Asia)
6. **Emotion** of the speaker (Happy/Sad/Anger/Neutral/Frustrated)
