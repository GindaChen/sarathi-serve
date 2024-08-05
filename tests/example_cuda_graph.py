import datetime
from sarathi.config.config import WorkerConfig
from tqdm import tqdm
from typing import List

from sarathi.config import ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig, SystemConfig, \
    ReplicaConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
import argparse

from sarathi.types import AttentionBackend

BASE_OUTPUT_DIR = "./offline_inference_output"

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def parse_args():
    parser = argparse.ArgumentParser("Static Sarathi serve with CUDA Graph enabled.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7B-hf")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()
    return args


args = parse_args()

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model=args.model,
    # load_format="dummy",
)

parallel_config = ParallelConfig(
    tensor_parallel_size=args.tp,
    pipeline_parallel_size=args.pp,
)

scheduler_config = SarathiSchedulerConfig(
    chunk_size=args.chunk_size,
    max_num_seqs=args.max_num_seqs,
)

metrics_config = MetricsConfig(
    write_metrics=True,
    # enable_chrome_trace=True,
)

worker_config = WorkerConfig(
    attention_backend=AttentionBackend.FLASHINFER_CUDA,
)

system_config = SystemConfig(
    replica_config=replica_config,
    model_config=model_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    metrics_config=metrics_config,
    worker_config=worker_config,
)


def generate(
    llm_engine: LLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[RequestOutput]:
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    # Run the engine
    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)

    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=args.max_tokens)


def test_generate(prompts):
    outputs = generate(llm_engine, prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.text
        print("===========================================================")
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("===========================================================")



llm_engine = LLMEngine.from_system_config(system_config)
print(system_config)

for seq_len in [8, 42, 128,]:
    for batch_size in [1, 2, 3, 4, 6, 8]:
        print(f"Testing seq_len={seq_len}, batch_size={batch_size}")
        prompts = [" a" * seq_len] * batch_size
        test_generate(prompts)
        print(f"Test success: seq_len={seq_len}, batch_size={batch_size}")

    # llm_engine.pull_worker_metrics()
# llm_engine.plot_metrics()
