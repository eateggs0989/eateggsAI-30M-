import sys
import os
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time
import winsound

from model.model import GPT, GPTConfig

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading training data...")
blockfile = os.path.join(BASE_DIR,"dataset","train_blocks.pt")
data = torch.load(blockfile)

# Shifted inputs/targets
x = data[:, :-1]
y = data[:, 1:]

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=11, shuffle=True)

# -----------------------------
# Initialize Model
# -----------------------------
config = GPTConfig()
model = GPT(config).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Mixed precision scaler
scaler = GradScaler()

# -----------------------------
# Training Loop
# -----------------------------
epochs = 3
total_steps = len(loader) * epochs
global_step = 0
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_start = time.time()

    for step, (xb, yb) in enumerate(loader):
        step_start = time.time()

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        # ---- Mixed Precision Forward ----
        with autocast():
            logits, loss = model(xb, yb)

        # ---- Backward ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        global_step += 1

        # ---- Metrics ----
        step_time = time.time() - step_start
        elapsed = time.time() - start_time
        steps_left = total_steps - global_step
        eta = steps_left * step_time
        current_lr = optimizer.param_groups[0]["lr"]

        if step % 50 == 0:
            print(
                f"Epoch {epoch+1} | Step {step}/{len(loader)} | "
                f"Loss {loss.item():.4f} | "
                f"LR {current_lr:.6f} | "
                f"{step_time:.2f}s/step | "
                f"ETA {eta/60:.1f} min"
            )

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch+1} completed.")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Epoch Time: {epoch_time/60:.2f} minutes\n")

    # 🔔 Beep after each epoch
    winsound.Beep(1000, 500)

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "gpt_6layer.pt")
print("Model saved as gpt_6layer.pt")

# Final sound
winsound.Beep(1500, 800)
                        