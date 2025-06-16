import pandas as pd

root = "/home/tthebau1/EDART/SpeechLLM/data/"

sets = {
    "train":['librispeech_train-clean-100', 'iemocap_ses01',  'iemocap_ses02',  'iemocap_ses03', 'voxceleb1_dev'],
    "dev":['librispeech_dev-clean', 'iemocap_ses04', 'voxceleb1_test'],
    "test":['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test'],
}

for set in sets:
    df = pd.read_csv(root+sets[set][0]+".csv")
    for data in sets[set][1:]:
        add = pd.read_csv(root+data+".csv")
        df = pd.concat([df, add], axis=0)
    
    df.to_csv(root+set+'.csv', index=False)
    print(f"saved {set} set in {root}{set}.csv, shape={df.shape}")
