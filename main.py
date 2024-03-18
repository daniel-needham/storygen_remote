from vllm import LLM, SamplingParams
import pandas as pd

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

sampling_params = SamplingParams(max_tokens=4096, # set it same as max_seq_length in SFT Trainer
                  temperature=0.2,
                  skip_special_tokens=True)

#import book paragraphs from pkl file
df = pd.read_pickle('data/ghost_stories/tokened_data/top10books_50_50.pkl')
# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
# shorten the data for testing
#df = df.head(10)
input_data = df['paragraph'].tolist()
prompts = []

TEMPLATE = """[INST]You are a creative writing teacher specialising in the genre of ghost stories. Using the provided context: '{text_chunk}', formulate a clear writing prompt backwards from the context. For example: 'Write a narrative section from a 3rd person perspective, involving a young warrior lost in a cave, hunted by primitive men, saved by the glow of a magical artifact he carries.' or 'Write a narrative section where a character named Steven opens a window of his home, letting in a stiff wind and maybe some form of spectral being.'. Begin the prompt with 'Write a narrative section'. Avoid numbering and make the prompt a succinct overview of the piece of text in a maximum of 2 sentences.[/INST]"""


def add_prompt(text_chunk):
    prompt = TEMPLATE.format(text_chunk=text_chunk)
    return prompt


for text_chunk in input_data:
    text = add_prompt(text_chunk)
    prompts.append(text)


outputs = llm.generate(prompts, sampling_params)  # Batch inference


df['prompt'] = prompts
df['generated_text'] = [output.outputs[0].text for output in outputs]
df['generated_text_length'] = df['generated_text'].apply(len)

df.to_pickle('data/ghost_stories/fine_tuning_prompts/top10books_500prompts.pkl')
df.to_csv('data/ghost_stories/fine_tuning_prompts/top10books_500prompts.csv')