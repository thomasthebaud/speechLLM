from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

def get_llm(name, use_lora, lora_r, lora_alpha):
    if 'mistralai' in name: 
        llm_tokenizer, llm_model = get_mistral(name)
    else:
        print('Loading a generic LLM with AutoModelForCausalLM')
        llm_tokenizer = AutoTokenizer.from_pretrained(name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            name, 
            trust_remote_code=True,
            )

    if use_lora:
        print('Adding LoRas')
        peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear",
                lora_dropout=0.05,
            task_type="CAUSAL_LM",
            )

        llm_model = get_peft_model(llm_model, peft_config)
        llm_model.print_trainable_parameters()
    else:
        print('No LoRa used, freezing parameters of LLM')
        for param in llm_model.parameters():
            param.requires_grad = False
        llm_model.train()

    return llm_tokenizer, llm_model

def get_mistral(name):
    local_dir = '/export/fs05/tthebau1/EDART/HF_models/Ministral-3-3B'
    print(f"Loading a Mistral model ({name}) locally from {local_dir}")
    llm_model = Mistral3ForConditionalGeneration.from_pretrained(local_dir, local_files_only=True)
    print("model Loaded, Loading a Mistral tokenizer")
    llm_tokenizer = MistralCommonBackend.from_pretrained(local_dir, local_files_only=True)

    return llm_tokenizer, llm_model