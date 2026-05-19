import sys
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from model.model import GPT, GPTConfig

version = "v0.1"

print("=" * 60)
print("         eateggsAI-30M", version)
print("    Experimental GPT-Style LLM")
print("=" * 60)
# -------------------------
# Device
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device     : {device}")


# -------------------------
# Load tokenizer
# -------------------------

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# -------------------------
# Load model
# -------------------------

config = GPTConfig()

model = GPT(config).to(device)

print("Loading model weights...")
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "gpt_6layer.pt")
model.load_state_dict(
    torch.load(checkpoint_path , map_location=device)
)

model.eval()

print("Model loaded.\n")


# -------------------------
# Generation settings
# -------------------------

temperature = 0.5 #0.8 for creativity we need correct answer so we use 0.5
top_k = 20  #40 for random answer now we don't too much random so we do 20
top_p = 0.9
repetition_penalty = 1.1 #1.15we reduce this to 1.1 
max_new_tokens = 120


# -------------------------
# Generate function
# -------------------------

def generate(prompt):

    tokens = tokenizer.encode(prompt)

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated = tokens.tolist()[0]

    for _ in range(max_new_tokens):

        # context clipping
        if tokens.size(1) > config.block_size:
            tokens = tokens[:, -config.block_size:]

        with torch.no_grad():
            logits, _ = model(tokens)

        logits = logits[:, -1, :] / temperature


        # -------------------------
        # repetition penalty
        # -------------------------

        for token in generated[-50:]:
            logits[0, token] /= repetition_penalty


        # -------------------------
        # top-k filtering
        # -------------------------

        if top_k > 0:

            values, _ = torch.topk(logits, top_k)

            min_values = values[:, -1].unsqueeze(-1)

            logits[logits < min_values] = -float("Inf")


        # -------------------------
        # probabilities
        # -------------------------

        probs = F.softmax(logits, dim=-1)


        # -------------------------
        # top-p nucleus sampling
        # -------------------------

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_probs[sorted_indices_to_remove] = 0

        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)


        next_token = torch.multinomial(sorted_probs, 1)

        next_token = sorted_indices.gather(-1, next_token)

        token_id = next_token.item()

        generated.append(token_id)

        tokens = torch.cat([tokens, next_token], dim=1)


    return tokenizer.decode(generated, skip_special_tokens=True)


# -------------------------
# Interactive loop
# -------------------------
total_params = sum(p.numel() for p in model.parameters())



while True:

    prompt = input("\nEnter prompt: ")

    if prompt.strip() == "":
        continue

    output = generate(prompt)

    print("\n[ eateggsAI Generating Response... ]\n")
    print(output)
