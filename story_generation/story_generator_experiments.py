from utils import load_text_file
from vllm import LLM, SamplingParams

story_structure = load_text_file('story_structure.txt')


llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

sampling_params = SamplingParams(max_tokens=4096, # set it same as max_seq_length in SFT Trainer
                  temperature=0.2,
                  skip_special_tokens=True)

PROMPT_TEMPLATE = """[INST]You are a renowned writer specialising in the genre of {}. Using the [Story Structure] and the given [Story Events], create a numbered structure with descriptions of each of the events. Try to preserve the ordering of the given [Story Events] in the generated story structure.\n{}\n{}[/INST]"""

events = ["[Story Events]"] + ["John has a large heart", "john meets mary", "john and mary meet john's ex girlfriend", " a love triangle forms", "john dies"] + ["[/Story Events]"]

events = "\n".join(events)

print(events)

genre = "love stories"

prompt = PROMPT_TEMPLATE.format(genre, story_structure, events)

output = llm.generate(prompt, sampling_params)

print(output)
print(output[0].outputs)
