import os

def load_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

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
                # remove all text before "[/INST]" including "[/INST]"
                ind = generated_text.find("[/INST]")
                generated_text = generated_text[(ind + len("[/INST]")):]
                # remove "</s>"
                generated_text = generated_text.replace("</s>", "")

                generated_text = generated_text.strip()

                # save the cleaned text
                with open(filename, 'w') as f:
                    f.write(generated_text)