import fire
import json
import os
import yaml
import datasets
import tiktoken
from datasets import load_dataset
from pathlib import Path
from bench_utils import (
    load_model_answers,
    model_name_to_id
)    
from icecream import ic
from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import base64
workers = []

worker_initiated = False
default_generation_config = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 4096,
}
def encode_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return json.dumps(im_64)

def image_to_url(image:Image.Image, image_format="PNG") -> str:
    image_format = image.format.lower() if image.format else image_format
    return f"data:image/{image_format};base64,{encode_image(image, image_format=image_format)}"


def main(
    dataset_path: str="WildVision/wildvision-bench",
    dataset_name: str="vision_bench_0617",
    dataset_split: str="test",
    worker_addr: str=None,
    model_name: str="mistralai/Pixtral-12B-2409",
    tokenizer_mode: str="auto",
    results_dir: str=None,
    LOG_DIR="./logs",
    bench_name="vision_bench_0617",
    num_gpu: int=1,
    max_model_len: int=None,
):
    """
    Args:
        dataset_path: the path to the dataset
        dataset_name: the name of the dataset
        dataset_split: the split of the dataset to use
        worker_addr: the address of the worker to use
        model_name: the name of the model to launch, huggingface model name
        LOG_DIR: the directory to save the logs
        
        if worker_addr is provided, the model will be launched on the worker_addr
        if worker_addr is not provided, the model will be launched locally
        At least one of worker_addr or model_name must be provided
    """
    assert model_name is not None or worker_addr is not None, "Either model_name or worker_addr must be provided"
    if results_dir is None:
        results_dir = f"data/{bench_name}/model_answers/"
        
    os.environ["WILDVISION_ARENA_LOGDIR"] = LOG_DIR 
    
    # try load existing generation configs
    config_yml_path = Path(os.path.abspath(__file__)).parent / "model_configs" / model_name_to_id(model_name) / "generation_config.yml"
    if config_yml_path.exists():
        print(f"Loading existing config from {config_yml_path}")
        with open(config_yml_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"No existing model specific config found for {model_name}")
        print("Creating new default config based on default_generation_config: {}", default_generation_config)
        
        config_yml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_yml_path, "w") as f:
            yaml.dump(default_generation_config, f)
        print(f"Created new default config at {config_yml_path}")
        config = default_generation_config
        
    print(f"Loaded generation config: {config}")
    
    # Load the dataset
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    print(dataset)
    existing_answers = load_model_answers(results_dir)
    if model_name_to_id(model_name) in existing_answers:
        model_existing_answers = existing_answers[model_name_to_id(model_name)]
        
        def map_fill_existing(item):
            question_id = item['question_id']
            if question_id in model_existing_answers and not "ERROR" in model_existing_answers[question_id]['output']:
                print(model_existing_answers[question_id]['output'])
                item['output'] = model_existing_answers[question_id]['output']
                item['token_len'] = len(encoding.encode(item['output'], disallowed_special=()))
            else:
                item['output'] = None
                item['token_len'] = None
            return item
        dataset = dataset.map(map_fill_existing, writer_batch_size=200) # pretty weird bug, need to set writer_batch_size to avoid the mapping error
        print("Filled existing answers")
        to_generate_indices = [i for i, item in enumerate(dataset) if item['output'] is None]
    else:
        to_generate_indices = list(range(len(dataset)))
    
    print(f"Generating {len(to_generate_indices)} items for {model_name}")
    llm = LLM(model=model_name, tokenizer_mode=tokenizer_mode, tensor_parallel_size=num_gpu, max_model_len=max_model_len, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        max_tokens=config.get("max_new_tokens", 4096),
        top_p=config.get("top_p", 1.0),
        temperature=config.get("temperature", 0.0),
    )
    
    if len(to_generate_indices) == 0:
        print(f"No items to generate for {model_name}")
    else:
        # all_instructions = dataset['instruction']
        all_instructions = [dataset[i]['instruction'] for i in to_generate_indices]
        new_dataset = datasets.Dataset.from_dict({
            "instruction": all_instructions,
        })
        def process_messages(item, index):
            item['messages'] = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": item['instruction']}, {"type": "image_url", "image_url": {"url": image_to_url(dataset[index]['image'])}}]
                },
            ]
            item['messages'] = json.dumps(item['messages'])
            return item
        new_dataset = new_dataset.map(process_messages, num_proc=8, with_indices=True, desc="Processing messages")
        all_messages = new_dataset['messages']
        all_messages = [json.loads(x) for x in tqdm(all_messages, desc="Loading messages")]
        assert not any([x is None for x in all_messages]), "Some messages are None"
        # all_messages = new_dataset['messages']
        
        outputs = llm.chat(all_messages, sampling_params=sampling_params)
        all_outputs = [x.outputs[0].text for x in outputs]
        
        def map_assign_output(item, index):
            if index in to_generate_indices:
                item['output'] = all_outputs[to_generate_indices.index(index)]
                item['token_len'] = len(encoding.encode(item['output'], disallowed_special=()))
            item['model'] = model_name
            return item
        dataset = dataset.map(map_assign_output, num_proc=2, with_indices=True, desc="Assigning outputs", remove_columns=["image"])
        
        results_file = results_dir + model_name_to_id(model_name) + ".jsonl"
        
        with open(results_file, "w") as f:
            for i, item in enumerate(dataset):
                f.write(json.dumps(item) + "\n")
        # dataset.to_json(results_file, orient="records", lines=True)
        # new_dataset.save_to_disk(os.path.join(results_dir, model_name))
        print(f"Saved {model_name} answers to {results_file}")
    
    for worker in workers:
        worker.terminate()
    print("Done")



if __name__ == "__main__":
    fire.Fire(main)
