from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
# from huggingface_hub import snapshot_download

model_id = "mistralai/Ministral-3-3B-Base-2512"
print('loading model')
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = MistralCommonBackend.from_pretrained(model_id)
print("model loaded")

local_dir = '/export/fs05/tthebau1/EDART/HF_models/Ministral-3-3B'
print("saving model")
model.save_pretrained(local_dir, safe_serialization=True, max_shard_size="2GB")
print("model saved, saving tokenizer")
tokenizer.save_pretrained(local_dir)
print('tokenizer saved')


# snapshot_download(
#     repo_id=model_id,
#     local_dir=local_dir,
#     max_workers=1
# )