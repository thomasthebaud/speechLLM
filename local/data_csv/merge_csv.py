import pandas as pd

root = "/home/tthebau1/EDART/SpeechLLM/data/"

sets = {
    "train":['librispeech_train-clean-100', 'iemocap_ses01',  'iemocap_ses02',  'iemocap_ses03', 'voxceleb1_dev', 'CV-EN_train', 'MSP_Podcast_Train'],
    "dev":['librispeech_dev-clean', 'iemocap_ses04', 'voxceleb1_test', 'CV-EN_dev', 'MSP_Podcast_Validation'],
    "test":['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test', 'CV-EN_test', 'MSP_Podcast_Test'],
}

for set in sets:
    df = pd.read_csv(root+sets[set][0]+".csv")
    for data in sets[set][1:]:
        add = pd.read_csv(root+data+".csv")
        df = pd.concat([df, add], axis=0)
    
    df.to_csv(root+set+'.csv', index=False)
    print(f"saved {set} set in {root}{set}.csv, shape={df.shape}")
