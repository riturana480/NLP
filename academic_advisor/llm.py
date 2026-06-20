"""Shared LLM. Loads Qwen2.5-3B-Instruct ONCE and exposes a simple chat() helper.

Both the router and every agent call chat(), so the 6 GB model is loaded a single
time and reused across the whole system.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LLM_MODEL, MAX_NEW_TOKENS

_tokenizer = None
_model = None


def _load():
    """Lazy-load the tokenizer + model on first use."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading LLM: {LLM_MODEL} ...")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            dtype="auto",       # was `torch_dtype` (renamed in recent transformers)
            device_map="auto",  # uses GPU if available (needs `accelerate`)
        )
        print("LLM ready on:", _model.device)
    return _tokenizer, _model


def chat(system, user, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0):
    """Run one chat turn through the instruct model. Returns the reply text.

    temperature=0.0 -> deterministic (greedy), which is what we want for both
    routing and grounded answers.
    """
    tokenizer, model = _load()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    # Strip the prompt tokens, keep only what the model generated.
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
