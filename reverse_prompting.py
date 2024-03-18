from vllm import LLM, SamplingParams
import pandas as pd

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

sampling_params = SamplingParams(max_tokens=4096, # set it same as max_seq_length in SFT Trainer
                  temperature=0.2,
                  skip_special_tokens=True)

#import book examples
df = pd.read_csv('data/science_fiction/science_fiction_tiling.csv')
# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
# shorten the data for testing
#df = df.head(10)
input_data = df['example'].tolist()
prompts = []

TEMPLATE_GS = """[INST]You are a creative writing teacher specialising in the genre of ghost stories. Using the provided context: '{text_chunk}', formulate a clear writing prompt backwards from the context. For example: 'Write a narrative section from a 3rd person perspective, involving a young warrior lost in a cave, hunted by primitive men, saved by the glow of a magical artifact he carries.' or 'Write a narrative section where a character named Steven opens a window of his home, letting in a stiff wind and maybe some form of spectral being.'. Begin the prompt with 'Write a narrative section'. Avoid numbering and make the prompt a succinct overview of the piece of text in a maximum of 2 sentences.[/INST]"""
TEMPLATE_LS = """[INST]You are a creative writing teacher specialising in the genre of love stories. Using the provided context: '{text_chunk}', formulate a clear writing prompt backwards from the context. For example: 'Write a narrative section introducing the character Jane, who goes on to have a scandalous conversation with the new groundskeeper' or 'Write a narrative section where a character named Steven opens a window of his stately home, noticing that his wife is talking with a group of mean he does not recognise'. Begin the prompt with 'Write a narrative section'. Avoid numbering and make the prompt a succinct overview of the piece of text in a maximum of 2 sentences.[/INST]"""
TEMPLATE_SF = """[INST]You are a creative writing teacher specialising in the genre of science fiction. Using the provided context: '{text_chunk}', formulate a clear writing prompt backwards from the context. For example: 'Write a narrative section from a 3rd person perspective, where a young man is met by a humanoid robot seeking a meeting with the Captain of his ship' or 'Write a narrative section where the character Mooney finds out that his time onboard the planet has been a facade, with the inhabitants seemingly knowing that he has come from offworld'. Begin the prompt with 'Write a narrative section'. Avoid numbering and make the prompt a succinct overview of the piece of text in a maximum of 2 sentences.[/INST]"""
def add_prompt(text_chunk):
    prompt =TEMPLATE_SF.format(text_chunk=text_chunk)
    return prompt


for text_chunk in input_data:
    text = add_prompt(text_chunk)
    prompts.append(text)


outputs = llm.generate(prompts, sampling_params)  # Batch inference


df['prompt'] = prompts
df['generated_text'] = [output.outputs[0].text for output in outputs]
df['generated_text_length'] = df['generated_text'].apply(len)

df.to_pickle('data/science_fiction/fine_tuning_prompts/science_fiction_ft_ds.pkl')
df.to_csv('data/science_fiction/fine_tuning_prompts/science_fiction_ft_ds.csv')