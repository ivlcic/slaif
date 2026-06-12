import torch

from transformers import pipeline

torch.set_default_device("cuda")

def main():
    p = pipeline("text-generation",
                        model="Qwen/Qwen3-4B-Instruct-2507",
                        dtype=torch.float16, device_map="auto")

    chat = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'My internet keeps disconnecting'}
    ]

    for _ in range(3):
        response = p(chat, max_new_tokens=64, do_sample=True, top_k=10)
        print(response[0]["generated_text"][-1]["content"])

    print("Hello from slaif!")


if __name__ == "__main__":
    main()
