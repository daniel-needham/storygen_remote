from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput, LLM
from vllm.lora.request import LoRARequest
from typing import Optional, List, Tuple

class GenEngine:

    def __init__(self):
        self.engine = self.initialize_engine()


    def initialize_engine() -> LLMEngine:
        """Initialize the LLMEngine."""
        # max_loras: controls the number of LoRAs that can be used in the same
        #   batch. Larger numbers will cause higher memory usage, as each LoRA
        #   slot requires its own preallocated tensor.
        # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
        #   numbers will cause higher memory usage. If you know that all LoRAs will
        #   use the same rank, it is recommended to set this as low as possible.
        # max_cpu_loras: controls the size of the CPU LoRA cache.
        engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                                 enable_lora=True,
                                 max_loras=1,
                                 max_lora_rank=16,
                                 max_cpu_loras=2,
                                 max_num_seqs=256)
        return LLMEngine.from_engine_args(engine_args)

    def process_requests(engine: LLMEngine,
                         test_prompts: List[Tuple[str, SamplingParams,
                         Optional[LoRARequest]]]):
        """Continuously process a list of prompts and handle the outputs."""
        request_id = 0

        while test_prompts or engine.has_unfinished_requests():
            if test_prompts:
                prompt, sampling_params, lora_request = test_prompts.pop(0)
                engine.add_request(str(request_id),
                                   prompt,
                                   sampling_params,
                                   lora_request=lora_request)
                request_id += 1

            request_outputs: List[RequestOutput] = engine.step()

            for request_output in request_outputs:
                if request_output.finished:
                    print(request_output)