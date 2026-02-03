import pandas as pd

root = "/home/tthebau1/EDART/SpeechLLM/data/"

for csv in ["dev", "dev_sw", "switchboard_val"]:
    df = pd.read_csv(root+csv+".csv")
    output = []
    for s in ['voxceleb2_enriched', 'librispeech', 'MSP_Podcast']:#set(df['dataset']):
        df = df[df['dataset']==s]
        if s=='MSP_Podcast': df = df[df['emotion'].isin(['Happy', 'Angry', 'Neutral', 'Sad'])]
        if len(df)>100:out = df.sample(n=100)
        if len(output)==0: output=out
        else: output = pd.concat((output, out))

    output.to_csv(root+csv+"_xs.csv", index=False)
