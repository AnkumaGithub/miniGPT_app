from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from peft import PeftModel

# 1. Загрузите токенизатор с добавленными токенами
tokenizer = GPT2TokenizerFast.from_pretrained("./results/checkpoint-441")

# 2. Загрузите базовую модель и измените размер эмбеддингов
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
base_model.resize_token_embeddings(len(tokenizer))  # Исправляет размер mismatch

# 3. Загрузите LoRA-адаптеры
model = PeftModel.from_pretrained(base_model, "./results/checkpoint-441")
model = model.merge_and_unload()

# 4. Генерация текста
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1
)

prompt = "Once upon a time,"
result = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(result[0]["generated_text"])