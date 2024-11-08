from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# from huggingface_hub import login

# login(token="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")

# Configuration Settings
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_NAME = "openai-community/gpt2-large"
    # TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    # MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

