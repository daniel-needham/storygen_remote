
from llm_engine import *
from genre import Genre

## run test
draft_prompt = """[INST] You are a renowned creative writer specialising in the genre of science fiction. Write a narrative section in the third person, where a man discovers a locked book, which may contain answers to the mystery of the stately home he is currently staying in. There is a strange chill in the air, as the suspicious maid enters the parlour [/INST]"""

# Load the model
gen = GenEngine()


sampling_params = SamplingParams(max_tokens=4096,
                                         temperature=0.7,
                                         top_k=50,
                                         repetition_penalty=0.5,
                                         frequency_penalty=1.19,
                                         )

# Generate the story
output = gen.process_requests([(draft_prompt, sampling_params, Genre.SCIENCE_FICTION)])
output = output[0].outputs[0].text

print(output)