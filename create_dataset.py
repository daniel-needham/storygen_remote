import os
import pandas as pd
import shutil
import nltk
import numpy as np
from tqdm import tqdm
from utils import load_text_file
import re

nltk.download('punkt')


def dynamic_print(message):
    print(message, end='\r')
    # Flush the output buffer to ensure the message is displayed immediately
    print('', end='', flush=True)

def clean_text(para):
    # para = [s.replace('\n', ' ') for s in para]
    para = [s.replace('_', '') for s in para]
    para = [s.replace(' *       *       *       *       * ', '') for s in para]
    para = [s.replace('*       *       *       *       *', '') for s in para]
    para = [s.replace('[Illustration]', '') for s in para]

    return para

def clean_text_single(para):
    pattern = r'^\n\n([A-Z0-9\s.,;:!?-]+)\n\n'
    # Find all matches
    matches = re.findall(pattern, para, flags=re.MULTILINE)
    print(f'Found {len(matches)} chapter headings. Removing them.')
    para = re.sub(pattern, '\n\n', para, flags=re.MULTILINE)
    para = para.replace('_', '')
    para = para.replace(' *       *       *       *       * ', '')
    para = para.replace('*       *       *       *       *', '')
    para = para.replace('[Illustration]', '')
    para = para.strip()

    return para

def remove_invalid_excerpts(paragraphs, excerpts_idx_tuples):
    valid_excerpts = []
    for i, size in excerpts_idx_tuples:
        # check if the excerpt is not out of bounds and if it does not intersect with any other excerpt
        if i + size < len(paragraphs) and not any(i <= start < i + size for start, _ in valid_excerpts):
            valid_excerpts.append((i, size))
    print(f"Removed {len(excerpts_idx_tuples) - len(valid_excerpts)} invalid excerpts")
    return valid_excerpts

def delete_unused_books(file_path):
    df = pd.read_csv(file_path + "/metadata.csv")
    # create new folder
    new_folder = file_path + "/text2"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for index, row in df.iterrows():
        # copy file to new location
        file_name = row['id'] + '_text.txt'
        sub_file_path = os.path.join(file_path, "text", file_name)
        # copy file from old location to new location
        new_file_path = os.path.join(new_folder, file_name)
        shutil.copy(sub_file_path, new_file_path)

    # delete old folder
    shutil.rmtree(file_path + "/text")
    os.rename(new_folder, file_path + "/text")


def create_dataset(file_path, paragraph_amount_range=(10,20), excerpts_per_book=10):
    df = pd.read_csv(file_path + "/metadata.csv")

    book_id = []
    indices = []
    examples = []

    for index, row in df.iterrows():
        file_name = row['id'] + '_text.txt'
        sub_file_path = os.path.join(file_path, "text", file_name)
        book = load_text_file(sub_file_path)

        #tokenize the book into paragraphs
        sentences = nltk.sent_tokenize(book)
        paragraphs = []
        current_paragraph = []

        for sentence in sentences:
            current_paragraph.append(sentence)

            if sentence.endswith(('.', '!', '?', '\n')):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []

            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))

        for i in range(excerpts_per_book):
            paragraph_size = np.random.randint(*paragraph_amount_range)
            start = np.random.randint(0, len(paragraphs) - paragraph_size)
            end = start + paragraph_size
            para = paragraphs[start:end].copy()
            para = [s for s in para if len(s) > 1]
            #para = [s.replace('\n', ' ') for s in para]
            para = [s.replace('_', ' ') for s in para]
            para = [s.replace(' *       *       *       *       * ', '') for s in para]
            para = [s.replace('*       *       *       *       *', '') for s in para]

            book_id.append(row['id'])
            indices.append((start, end))
            examples.append('\n'.join(para))

    return pd.DataFrame({'book_id': book_id, 'indices': indices, 'example': examples})

def create_dataset_tiling(file_path, force_tokenize, n_examples=1500, target_chunk_size=300):
    df = pd.read_csv(file_path + "/metadata.csv")

    book_id = []
    indexes = []
    examples = []
    lengths = []

    for index, row in tqdm(df.iterrows()):
        file_name = row['id'] + '_text.txt'
        text_file_path = os.path.join(file_path, "text", file_name)
        tokenized_file_path = os.path.join(file_path, "tokenized", row['id'] + '_tokenized.pkl')


        if not os.path.exists(tokenized_file_path) or force_tokenize:
            # load the book
            book = load_text_file(text_file_path)

            # tokenize the book into sentences
            print('Sentence tokenization')
            sentences = nltk.sent_tokenize(book)

            print('Calculating average sentence length')
            sentence_lengths = [len(nltk.word_tokenize(p)) for p in sentences]
            sentence_avg = np.mean(sentence_lengths)
            print(f"Average sentence length: {sentence_avg}")
            print(f"Number of sentences: {len(sentences)}")

            if len(sentences) < target_chunk_size:
                print(f"Book {row['id']} has less than {target_chunk_size}. Setting target_chunk_size to {len(sentences)}")
                w= len(sentences)
            else:
                w = target_chunk_size

            k = int(0.5 * w)
            print(f"Using w={w}, k={k} for text tiling.")
            ttt = nltk.TextTilingTokenizer(w=w, k=k)

            # tokenize the book into paragraphs
            book = clean_text_single(book)
            print(f'Text tiling {row["id"]}...')
            paragraphs = ttt.tokenize(book)
            pd.to_pickle(paragraphs, tokenized_file_path)

        else:
            paragraphs = pd.read_pickle(tokenized_file_path)


        paragraphs = [p for p in paragraphs if len(p) > 1]

        for idx, para in enumerate(paragraphs):
            book_id.append(row['id'])
            indexes.append(idx)
            examples.append(para)
            lengths.append(len(para))


    df = pd.DataFrame({'book_id': book_id, 'index': indexes, 'example': examples, 'length': lengths})

    # filter the dataframe to only contain samples with a length of at least 200
    df = df[df['length'] > 900]

    if len(df) < n_examples:
        print(f"Only {len(df)} examples available, returning all of them.")
    else:
        print(f"Returning {n_examples} examples from a total of {len(df)}.")

    return df.sample(n=min(n_examples, len(df)))

df = create_dataset_tiling('data/science_fiction', force_tokenize=True, n_examples=1500, target_chunk_size=50)
df.to_csv('data/science_fiction/science_fiction_tiling.csv', index=False)
