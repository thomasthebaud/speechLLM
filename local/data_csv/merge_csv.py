import pandas as pd

root = "/home/tthebau1/EDART/SpeechLLM/data/"

sets = {
    "train":['librispeech_train-clean-100', 'iemocap_ses01',  'iemocap_ses02',  'iemocap_ses03', 'voxceleb1_dev', 'CV-EN_train', 'MSP_Podcast_Train', 'voxceleb2_enriched_dev'],
    "dev":['librispeech_dev-clean', 'iemocap_ses04', 'voxceleb1_test', 'CV-EN_dev', 'MSP_Podcast_Validation', 'voxceleb2_enriched_test'],
    "test":['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test'],
    "train_sw":['librispeech_train-clean-100', 'iemocap_ses01',  'iemocap_ses02',  'iemocap_ses03', 'voxceleb1_dev', 'CV-EN_train', 'MSP_Podcast_Train', 'voxceleb2_enriched_dev', 'switchboard_train'],
    "dev_sw":['librispeech_dev-clean', 'iemocap_ses04', 'voxceleb1_test', 'CV-EN_dev', 'MSP_Podcast_Validation', 'voxceleb2_enriched_test', 'switchboard_val'],
    "test_sw":['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test', 'switchboard_test'],
}

for set in sets:
    df = pd.read_csv(root+sets[set][0]+".csv")
    for data in sets[set][1:]:
        add = pd.read_csv(root+data+".csv")
        if len(add.columns) != len(df.columns):
            if len(add.columns)>len(df.columns):
                for col in add.columns:
                    if col not in df.columns:
                        df[col]=None
            else:
                for col in df.columns:
                    if col not in add.columns:
                        add[col]=None
        assert len(add.columns)==len(df.columns)
        df = pd.concat([df, add], axis=0)
    
    df.to_csv(root+set+'.csv', index=False)
    print(f"saved {set} set in {root}{set}.csv, shape={df.shape}")
