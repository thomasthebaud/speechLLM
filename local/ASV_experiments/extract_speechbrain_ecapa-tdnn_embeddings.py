import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import pandas as pd
from tqdm import tqdm
import numpy as np

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

for input_set, output in [
    ('data/voxceleb1_test.csv', '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/all/'),
    ('data/voxceleb1_dev.csv', '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/all/')
]:
    audios= pd.read_csv(input_set)['audio_path']
    ids = ['-'.join(a.strip('.wav').split('/')[-3:]) for a in audios]
    for a,i in tqdm(zip(audios, ids)):
        signal, fs =torchaudio.load(a)
        print(signal.shape, fs)
        embeddings = classifier.encode_batch(signal)
        print(embeddings.shape)
        np.save(output+i, embeddings)
        break
    