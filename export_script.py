from datasets import load_dataset
import tqdm

x = load_dataset("wikipedia", language="simple", date="20220301")

with open('simple_wiki.txt', 'w') as f:
    for y in tqdm.tqdm(x['train']):
        f.write(y['text'] + '\n')
