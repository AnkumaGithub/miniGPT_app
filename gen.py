import argparse

import torch
import tiktoken
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50263,
    block_size=300,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.8,
    drop_path_rate=0.6,
    batch_size=12,
    lr=3e-4,
    bias=False,
    mode='ts_2',
    stride=300,
    weight_decay=0.1,
    pad_token_id=None
)

#CHECKPOINT_PATH = f"latest_checkpoint.pth"  # путь до чекпоинта
CHECKPOINT_PATH = f"E:\PyCharm 2024.3.5\projects\saves\_latest_checkpoint_{config.mode}.pth"  # путь до чекпоинта
ENCODING = "gpt2"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(config).to(DEVICE)

with torch.serialization.safe_globals([GPTConfig]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()

SPECIAL_TOKENS = ["[Q]", "[A]", "[SEP]", "[EOS]", "[USER]", "[BOT]"]

enc = tiktoken.get_encoding(ENCODING)
enc = tiktoken.Encoding(
    name=enc.name,
    pat_str=enc._pat_str,
    mergeable_ranks=enc._mergeable_ranks,
    special_tokens={**enc._special_tokens, **{token: len(enc._mergeable_ranks) + i for i, token in enumerate(SPECIAL_TOKENS)}}
)

def generate_text(prompt, max_new_tokens=200, temperature=0.3, top_p=0.85, repetition_penalty=8.0, stop_token="[EOS]"):
    input_ids = torch.tensor([enc.encode(prompt, allowed_special=set(SPECIAL_TOKENS))], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output, attention_maps = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            echo=True,
            stop_token=enc.encode(stop_token, allowed_special=set(SPECIAL_TOKENS))[0] if stop_token else None,
            bad_words_ids=[[enc.encode("[USER]", allowed_special=set(SPECIAL_TOKENS))[0]]]
        )

    tokens = [enc.decode([t]) for t in output[0].tolist()]

    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()
    prompt = input("\nпромт\n> ")
    output = generate_text(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("\nСгенерированный текст:\n")
    print("".join(output))

if __name__ == "__main__":
    main()

