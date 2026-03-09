import torchaudio
import torch
from speechbrain.inference.speaker import EncoderClassifier
import pandas as pd
from tqdm import tqdm
import numpy as np

if not torch.cuda.is_available():
    exit('cuda not available')

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

for input_set, output in [
    # ('data/voxceleb1_test.csv', '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/all/'),
    # ('data/voxceleb1_dev.csv', '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/all/'),
    # ('data/voxceleb2_test.csv', '/export/fs05/tthebau1/EDART/VoxCeleb2/ecapatdnn_speechbrain/all/'),
    ('data/voxceleb2_dev.csv', '/export/fs05/tthebau1/EDART/VoxCeleb2/ecapatdnn_speechbrain/all/'),
]:
    audios= pd.read_csv(input_set)['audio_path']
    ids = ['-'.join(a[:-4].split('/')[-3:]) for a in audios]
    for a,i in tqdm(zip(audios, ids), total=len(audios), desc=f'processing {input_set}'):
        signal, fs =torchaudio.load(a)
        # print(signal.shape, fs)
        embeddings = classifier.encode_batch(signal.to('cuda'))[0]
        # print(embeddings.shape)
        np.save(output+i, embeddings.detach().cpu().numpy())

    