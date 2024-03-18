from sentence_transformers import SentenceTransformer, util
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

sent_model = SentenceTransformer("all-MiniLM-L6-v2")

# load the data
val_set = pd.read_pickle('./data/metrics/val_ds.pkl')

# tokenize the paragraphs into sentences
val_set_sentences = []
[val_set_sentences.extend(nltk.sent_tokenize(par)) for par in val_set.paragraph]

# create val set embeddings
val_set_embeddings = sent_model.encode(val_set_sentences, convert_to_tensor=True)

# create a new dataframe to store the cosine similarity scores
cosine_similarity_df = pd.DataFrame(columns=['model', 'temp', 'prompt', 'cosine_similarity_avg', 'cosine_similarity_low', 'cosine_similarity_high'])

models = ['mb', 'ft']
temps = [0.6,0.7,0.8,0.9]
prompts = ['prompt1', 'prompt2', 'prompt3']
examples = [1,2,3,4,5]

for model in tqdm(models):
    for temp in temps:
        for prompt in prompts:
            temp_scores = []
            for i in examples:
                filename = """./data/metrics/examples/{}-{}-temp{}-{}.txt""".format(model, prompt, temp, i)
                generated_text = load_text_file(filename)
                # remove the prompt
                generated_text = generated_text[60:]
                # tokenize the generated text into sentences
                generated_text_sentences = nltk.sent_tokenize(generated_text)
                # create the generated text embeddings
                generated_text_embeddings = sent_model.encode(generated_text_sentences, convert_to_tensor=True)
                # calculate the cosine similarity scores
                temp_scores.append(util.cos_sim(val_set_embeddings, generated_text_embeddings).mean().item())

            cosine_similarity_df.loc[len(cosine_similarity_df)] = {'model': model, 'temp': temp, 'prompt': prompt, 'cosine_similarity_avg': np.mean(temp_scores), 'cosine_similarity_low': np.min(temp_scores), 'cosine_similarity_high': np.max(temp_scores)}

# pickle the cosine similarity dataframe
cosine_similarity_df.to_pickle('./data/metrics/cosine_similarity_df.pkl')
# # Compute embedding for both lists
# embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)
#
# # Compute cosine-similarities
# cosine_scores = util.cos_sim(embeddings1, embeddings2)
#
# # print average cosine score
# print('Average cosine similarity score:', cosine_scores.mean().item())