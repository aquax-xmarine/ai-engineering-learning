import time
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

inputs = tokenizer("The history of artificial intelligence began", return_tensors="pt")

def timed_generate(use_cache, new_tokens):
    start = time.time()
    model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tokenizer.eos_token_id,
    )
    return time.time() - start

timed_generate(True, 10)  # warm-up run

for n in [50, 200, 500]:
    with_cache = timed_generate(True, n)
    without_cache = timed_generate(False, n)
    print(f"{n} tokens | with cache: {with_cache:.2f}s | "
          f"without: {without_cache:.2f}s | speedup: {without_cache / with_cache:.1f}x")