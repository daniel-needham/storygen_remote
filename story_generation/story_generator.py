from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput, LLM
from vllm.lora.request import LoRARequest

from utils import *
import genre as import_genre
from genre import Genre
from ingest_story_json import ingest_story_json
import logging.config

# Load the logging configuration file
logging.config.fileConfig('logging.conf')

# Get a logger object
logger = logging.getLogger()

class StoryGenerator:
    def __init__(self):
        self.structure_template = load_text_file('structure_template.txt')
        self.PLOT_POINTS_AMT = 9
        self.PLOT_POINTS_NUMBERING = [("1.1", "Exposition"), ("1.2", "Inciting Incident") , ("1.3", "Plot Point A"), ("2.1", "Rising Action"), ("2.2", "Midpoint"), ("2.3", "Plot Point B"), ("3.1", "Pre Climax"), ("3.2", "Climax"), ("3.3", "Denouement")]
        self.id = None
        self.title = None
        self.genre = None
        self.premise = None
        self.characters = None
        self.plot_points = None

    def ingest_structure(self, structure_path: str):
        json = ingest_story_json(structure_path)
        self.id = json['id']
        self.genre = import_genre.create(json['genre'])
        self.premise = json['story_generation']['premise']
        self.characters = json['story_generation']['characters']
        self.plot_points = json['story_generation']['plot_points']

    def _get_prompt_ready_premise(self):
        return "Premise: " + self.premise
    def _plot_points_to_prompt(self):
        prompt = """
[Story Events]
Setup

1.1 {}
1.2 {}
1.3 {}

Confrontation

2.1 {}
2.2 {}
2.3 {}

Resolution

3.1 {}
3.2 {}
3.3 {}
[/Story Events]
"""
        values = []

        for plot_point_idx, _ in self.PLOT_POINTS_NUMBERING:
            if plot_point_idx in self.plot_points:
                values.append(self.plot_points[plot_point_idx]['description'])
            else:
                values.append("")

        prompt = prompt.format(*values)
        return prompt




        

    def generate_plot_points(self):
        llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

        sampling_params = SamplingParams(max_tokens=4096,
                                         # best_of=5,
                                         # set it same as max_seq_length in SFT Trainer
                                         temperature=0.6,
                                         # skip_special_tokens=True,
                                         # use_beam_search=True,
                                         stop=["\n"]
                                         )

        # if self.characters is None:
        #     # extract named entities from the plot points
        #     EXTRACT_NAMED_ENTITIES = """[INST]You are an expert at extracting names from text.
        #     Using the following example: "Mario is a large man for brooklyn who meets the love of his life daniella" \n1. Mario\n2. Daniella \n
        #     Do the same for the following text: \n{}[/INST]"""
        #
        #     ne_prompts = EXTRACT_NAMED_ENTITIES.format("\n".join(
        #         [plot_point["description"] for plot_point in self.plot_points]))
        #
        #     outputs = llm.generate(ne_prompts, sampling_params)
        #
        #     for output in outputs:
        #         prompt = output.prompt
        #         generated_text = output.outputs[0].text
        #         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        #
        #     ###todo turn the extracted named entities into characters

        # generate the plot points
        for plot_point_idx, plot_point_desc in self.PLOT_POINTS_NUMBERING:
            if plot_point_idx in self.plot_points:
                continue
            else:
                # generate the plot point
                current_plot_prompt = self._plot_points_to_prompt()
                plot_point_prompt = """[INST]You are a renowned writer specialising in the genre of {}. You are able to create engaging narratives following a three act structure. Using the [Story Structure] outline, fill the [Story Events] suitable for the story as outlined in the premise.\n{}{}{}\nCreate a single event for the plot point {}, keep it concise and avoid repeating previous plot points.[/INST] {} {}:"""
                plot_point_prompt = plot_point_prompt.format(self.genre.__str__(), self.structure_template, self._get_prompt_ready_premise(), current_plot_prompt, plot_point_idx, plot_point_idx, plot_point_desc)
                output = ""
                counter = 0
                while output == "":
                    counter += 1
                    output = llm.generate(plot_point_prompt, sampling_params)
                    output = output[0].outputs[0].text
                logger.info(f"Generated plot point {plot_point_idx} in {counter} attempts")
                # add the generated plot point to the plot points
                self.plot_points[plot_point_idx] = {"description": output.strip()}
                print(self._plot_points_to_prompt())




cl = StoryGenerator()
cl.ingest_structure('example.json')
cl.generate_plot_points()
