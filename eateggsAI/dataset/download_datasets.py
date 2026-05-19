
from datasets import load_dataset
from transformers import GPT2Tokenizer

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# -----------------------------
# Target token distribution (~20M total)
# -----------------------------
TARGETS = {
    "wiki": 12_000_000,
    "news": 3_000_000,
    "qa": 3_000_000,
    "books": 2_000_000
}

# -----------------------------
# Streaming with live progress
# -----------------------------
def stream_dataset(name, dataset, text_fn, target_tokens):
    texts = []
    token_count = 0
    last_print = 0

    print(f"\n--- Collecting {name} ---")

    for sample in dataset:

        try:
            text = text_fn(sample)
        except:
            text = sample

        if not text or not isinstance(text, str):
            continue

        if len(text) < 50:
            continue

        tokens = tokenizer.encode(text)
        token_count += len(tokens)
        texts.append(text.strip())

        # Print progress every 500k tokens
        if token_count - last_print >= 500_000:
            print(f"{name}: {token_count:,} tokens collected...")
            last_print = token_count

        if token_count >= target_tokens:
            break

    print(f"Finished {name}: {token_count:,} tokens.\n")
    return texts, token_count


total_tokens = 0

# -----------------------------
# Wikipedia
# -----------------------------
wiki = load_dataset(
    "wikipedia",
    "20220301.en",
    split="train",
    streaming=True
)

wiki_texts, t = stream_dataset(
    "Wikipedia",
    wiki,
    lambda x: x["text"] if isinstance(x, dict) else x,
    TARGETS["wiki"]
)
total_tokens += t


# -----------------------------
# News
# -----------------------------
news = load_dataset(
    "ag_news",
    split="train",
    streaming=True
)

news_texts, t = stream_dataset(
    "News",
    news,
    lambda x: x["text"] if isinstance(x, dict) else x,
    TARGETS["news"]
)
total_tokens += t


# -----------------------------
# Q&A
# -----------------------------
qa = load_dataset(
    "squad",
    split="train",
    streaming=True
)

qa_texts, t = stream_dataset(
    "Q&A",
    qa,
    lambda x: x["question"] + " " + x["context"]
    if isinstance(x, dict)
    else x,
    TARGETS["qa"]
)
total_tokens += t


# -----------------------------
# Books
# -----------------------------
books = load_dataset(
    "pg19",
    split="train",
    streaming=True
)

book_texts, t = stream_dataset(
    "Books",
    books,
    lambda x: x["text"] if isinstance(x, dict) else x,
    TARGETS["books"]
)
total_tokens += t


# -----------------------------
# Save
# -----------------------------
print("\nSaving dataset to file...")

all_texts = wiki_texts + news_texts + qa_texts + book_texts

with open("balanced_dataset_v2.txt", "w", encoding="utf-8") as f:
    for text in all_texts:
        f.write(text + "\n")

print("\nDataset v2 created successfully.")
print(f"Total tokens collected: {total_tokens:,}")
                        