import argparse
import os

os.environ["TMPDIR"] = "E:/temp_pytorch"
os.environ["TEMP"] = "E:/temp_pytorch"

import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("E:/PyCharm 2024.3.5/projects/results/checkpoint-1324")

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, "E:/PyCharm 2024.3.5/projects/results/checkpoint-1324")
    model = model.merge_and_unload()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    result = generator(
        args.prompt,
        max_length=args.max_tokens,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    print(result[0]["generated_text"].encode('utf-8', errors='replace').decode('utf-8'))

if __name__ == "__main__":
    main()