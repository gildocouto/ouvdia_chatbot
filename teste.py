#!/usr/bin/env python3
"""
test_llama3.py
Teste rápido do modelo Meta-Llama-3-8B-Instruct com transformers.
"""
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Carregando modelo (pode levar alguns minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()

messages = [
    {"role": "user", "content": "Quem é você? Explique suas funções"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(DEVICE)

print("Gerando resposta...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n--- Resposta do modelo ---")
print(reply)
