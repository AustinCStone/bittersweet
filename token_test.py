import tiktoken
import tqdm
from datasets import load_dataset
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
x = load_dataset("wikipedia", language="simple", date="20220301")
enc = tiktoken.encoding_for_model("gpt-4")
total_enc_len = 0
total_bytes = 0
for i, y in tqdm.tqdm(enumerate(x['train'])):
    encoding = enc.encode(y['text'])
    total_bytes += len(y['text'].encode('utf-8'))
    total_enc_len += len(encoding)
print(total_bytes / total_enc_len)


print(enc.encode("Hello how are you?"))