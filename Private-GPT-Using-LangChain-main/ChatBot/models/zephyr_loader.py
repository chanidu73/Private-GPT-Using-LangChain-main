
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_zephyr():
    model_id = "HuggingFaceH4/zephyr-7b-beta"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map ="auto",
        torch_dtype=torch.float32
    )

    pipe = pipeline("text-generation"  , model=model  , tokenizer = tokenizer)
    return pipe