import argparse

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("./results/checkpoint-441")

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, "./results/checkpoint-441")
    model = model.merge_and_unload()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if model.device.type == "cuda" else -1
    )

    result = generator(
        args.prompt,
        max_length=args.max_tokens,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()