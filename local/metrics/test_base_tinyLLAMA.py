import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

# Generation settings
MAX_NEW_TOKENS = 1048
TEMPERATURE = 0.2
TOP_P = 0.9

# Prompt used for each transcript
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that creates concise, faithful summaries of conversations."
)
USER_PROMPT_TEMPLATE = (
    "Summarize the following conversation in a brief paragraph. "
    "Capture participants, main topics, key decisions, and any explicit action items with owners and timelines. "
    "Avoid speculation and keep it under 120 words.\n\n"
    "Conversation:\n{transcript}"
)

df = pd.read_csv("data/switchboard_test.csv")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# FP16 if CUDA available, else CPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
)
model.eval()


def summarize_transcript(transcript: str) -> str:
    transcript = (transcript or "").strip()
    if not transcript:
        return ""

    user_prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)

    # Build chat messages and apply the model's chat template
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]

    # Convert to model input using chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = input_ids.to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # The newly generated portion is after the prompt length
    generated_ids = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return text.split("</s>")[0].strip()

# -----------------------------
# Run over the CSV
# -----------------------------
summaries = []
for transcript in tqdm(df['transcript'].tolist(), desc="Summarizing"):
    try:
        summary = summarize_transcript(transcript)
    except Exception as e:
        summary = f"[ERROR during summarization: {e}]"
    summaries.append(summary)

df["summary_out"] = summaries

print("done, now getting the scores")
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
rouge1_scores, rougeL_scores = [], []
for sys_sum, ref_sum in zip(df["summary_out"], df["summary"]):
    if pd.isna(ref_sum):
        rouge1_scores.append(None)
        rougeL_scores.append(None)
        continue
    scores = scorer.score(ref_sum, sys_sum)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

df["rouge1_f"] = rouge1_scores
df["rouge2_f"] = rouge2_scores
df["rougeL_f"] = rougeL_scores

print("Average ROUGE-1 F:", sum(s for s in rouge1_scores if s is not None) / len([s for s in rouge1_scores if s is not None]))
print("Average ROUGE-2 F:", sum(s for s in rouge2_scores if s is not None) / len([s for s in rouge2_scores if s is not None]))
print("Average ROUGE-L F:", sum(s for s in rougeL_scores if s is not None) / len([s for s in rougeL_scores if s is not None]))

