import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# 1) Load tokenizer + model
MODEL_NAME = "gpt2"
print_section(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# GPT-2 has no pad token by default; set pad to eos for generation APIs.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_demo(text: str):
    print_section("Tokenization Demo")
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("Input text:", text)
    print("Token IDs:", input_ids.tolist())
    print("Tokens:", tokens)
    print("Token count:", len(input_ids))

    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    print("Decoded back:", decoded)


def generate_text(prompt: str, max_new_tokens=60, temperature=1.0, top_k=50, top_p=1.0):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# 2) Tokenization experiments
text_examples = [
    "Hello world!",
    "I can't believe it's not butter.",
    "Transformers are amazing for NLP.",
    "Tokenization affects model behavior.",
]

for t in text_examples:
    tokenize_demo(t)


# 3) Decoding experiments
base_prompt = "Write a short motivational message for a student learning AI:"

settings = [
    {"temperature": 0.2, "top_k": 0, "top_p": 1.0},
    {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    {"temperature": 1.0, "top_k": 50, "top_p": 0.90},
    {"temperature": 1.3, "top_k": 100, "top_p": 0.95},
]

print_section("Decoding Strategy Experiments")
for i, s in enumerate(settings, 1):
    tk = s["top_k"]
    print(f"\n--- Run {i}: temp={s['temperature']} top_k={tk} top_p={s['top_p']} ---")
    out = generate_text(
        base_prompt,
        max_new_tokens=80,
        temperature=s["temperature"],
        top_k=tk,
        top_p=s["top_p"],
    )
    print(out)


# 4) Prompt sensitivity experiments
prompts = [
    "Summarize why practice matters in learning.",
    "Explain to a 10-year-old why practice matters in learning.",
    "In 3 bullet points, explain why practice matters in learning.",
]

print_section("Prompt Sensitivity Experiments")
for p in prompts:
    print("\nPrompt:", p)
    print(generate_text(p, max_new_tokens=70, temperature=0.8, top_k=50, top_p=0.95))


# 5) Completion-like vs instruction-like prompt
completion_prompt = "The future of AI in education is"
instruction_prompt = "Instruction: Give 3 practical ways AI can help students study better.\nAnswer:"

print_section("Completion vs Instruction-style Prompting")
print("\n[Completion prompt]")
print(generate_text(completion_prompt, max_new_tokens=80, temperature=0.9, top_k=50, top_p=0.95))

print("\n[Instruction-style prompt]")
print(generate_text(instruction_prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.95))

print_section("Done")
print("You now have a working LLM playground with tokenization and decoding controls.")
