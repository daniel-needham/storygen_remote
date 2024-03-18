from rank_bm25 import BM25Okapi
import pandas as pd
import re
import numpy as np

def load_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

# load the data
val_set = pd.read_pickle('./data/metrics/val_ds.pkl')
print(val_set.columns)


# create a new column to store the tokenized and cleaned text with no punctuation
val_set['text_clean'] = val_set.paragraph.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split()))

# create a concatenated list of all words in the corpus
tokenized_corpus = [doc.split(" ") for doc in val_set['text_clean']]
print(len(tokenized_corpus))

# create the bm25 model
bm25 = BM25Okapi(tokenized_corpus)

# create new dataframe to store the bm25 scores
bm25_df = pd.DataFrame(columns=['model', 'prompt', 'temp', 'bm25_avg_score', 'bm25_low_score', 'bm25_high_score'])
models = ['mb', 'ft']
temps = [0.6,0.7,0.8,0.9]
prompts = ['prompt1', 'prompt2', 'prompt3']
examples = [1,2,3,4,5]

for model in models:
    for temp in temps:
        for prompt in prompts:
            temp_scores = []
            for i in examples:
                filename = """./data/metrics/examples/{}-{}-temp{}-{}.txt""".format(model, prompt, temp, i)
                generated_text = load_text_file(filename)
                # clean and tokenize text
                clean_generated_text = [re.sub(r'[^a-zA-Z]','',w).lower() for w in generated_text.split()]
                scores = bm25.get_scores(clean_generated_text)
                temp_scores.append(np.mean(scores))
            bm25_df.loc[len(bm25_df)] = {'model': model, 'prompt': prompt, 'temp': temp, 'bm25_avg_score': np.mean(temp_scores), 'bm25_low_score': np.min(temp_scores), 'bm25_high_score': np.max(temp_scores)}

# pickle the bm25 dataframe
bm25_df.to_pickle('./data/metrics/bm25_df.pkl')
