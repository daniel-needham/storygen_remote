import nltk
import pandas as pd
def load_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def extract_named_entities(text):
    words = nltk.word_tokenize(text)
    #words = [word.capitalize() for word in words]
    tagged = nltk.pos_tag(words)
    print(tagged)

    for i,tag in enumerate(tagged):
        if tag[1].startswith('NNP'):
            words[i] = tag[0].capitalize()

    chunked = nltk.ne_chunk(tagged)

    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == nltk.tree.Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk