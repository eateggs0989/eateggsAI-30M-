from transformers import GPT2Tokenizer
import torch

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
block_size = 256
input_file = "balanced_dataset.txt"
output_file = "train_blocks.pt"
print("Reading dataset...")

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()
print("Tokenizing...")
tokens = tokenizer.encode(text)
print("Total tokens:", len(tokens))

# Drop extra tokens so it divides cleanly
total_length = (len(tokens) // block_size) * block_size
tokens = tokens[:total_length]

print("Creating blocks...")

blocks = []
for i in range(0, total_length, block_size):
    blocks.append(tokens[i:i+block_size])

# Convert to tensor
train_data = torch.tensor(blocks)

print("Total blocks:", train_data.shape[0])
print("Block shape:", train_data.shape)

# Save
torch.save(train_data, output_file)
print("Saved to", output_file)