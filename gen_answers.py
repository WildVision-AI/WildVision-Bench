import fire
import json
import requests
import subprocess
import random
import os
import time
import atexit
import yaml
import tiktoken
from functools import partial
from datasets import load_dataset
from pathlib import Path
from arena.model.model_adapter import get_conversation_template
from bench_utils import (
    chat_completion_openai,
    chat_completion_gemini,
    chat_completion_anthropic,
    chat_completion_rekaflash,
    chat_completion_idefics2,
    encode_image,
    load_model_answers,
    model_name_to_id
)    
from icecream import ic
workers = []

worker_initiated = False
default_generation_config = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 4096,
    'stop': None, 'stop_token_ids': None, 'echo': False
}

def call_api_worker_gpt(text, image, model_name, **generate_kwargs) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": image},
            ],
        }
    ]
    generated_text = ""
    if "gpt" in model_name:
        generated_text = chat_completion_openai(
            model_name,
            messages,
            temperature=generate_kwargs.get("temperature", 0.7),
            max_tokens=generate_kwargs.get("max_new_tokens", 2048),
        )
    elif "gemini" in model_name:
        generated_text = chat_completion_gemini(
            model_name,
            messages,
            temperature=generate_kwargs.get("temperature", 0.7),
            max_tokens=generate_kwargs.get("max_new_tokens", 2048), # TODO: remove according to this: https://github.com/google-gemini/generative-ai-python/issues/170#issuecomment-1890987296
        )
    elif "claude" in model_name:
        generated_text = chat_completion_anthropic(
            model_name,
            messages,
            temperature=generate_kwargs.get("temperature", 0.7),
            max_tokens=generate_kwargs.get("max_new_tokens", 2048),
        )
        ic(generated_text)
    elif "yi-vl-plus" in model_name:
        try:
            generated_text = chat_completion_openai(
                model_name,
                messages,
                temperature=generate_kwargs.get("temperature", 0.7),
                max_tokens=generate_kwargs.get("max_new_tokens", 2048),
                is_yivl_api=True,
            )
        except Exception as e:
            print(f"Error in yi-vl-plus API: {e}")
    elif "Reka-Flash" in model_name:
        model_name = "creeping-phlox-20240403"
        generated_text = chat_completion_rekaflash(
            model_name,
            text,
            image,
            temperature=generate_kwargs.get("temperature", 0.7),
            top_p=generate_kwargs.get("top_p", 1.0),
            max_tokens=generate_kwargs.get("max_new_tokens", 2048),
        )
    elif "idefics2-8b-chatty" in model_name:
        try:
            generated_text = chat_completion_idefics2(
                model_name,
                text,
                image,
                temperature=generate_kwargs.get("temperature", 0.7),
                top_p=generate_kwargs.get("top_p", 1.0),
                max_tokens=2048,#generate_kwargs.get("max_new_tokens", 2048),
            )
        except Exception as e:
            generated_text = "Error in idefics2, retrying..."
            raise e
            # time.sleep(5)
    else:
        raise ValueError("Unknown model name")
    
    
    print(generated_text)
    return generated_text

def call_local_worker(text, image, worker_addr, model_name, **generate_kwargs) -> str:
    global worker_initiated
    encoded_image = encode_image(image)
    conv_template = get_conversation_template(model_name)
    conv_template.messages = []
    conv_template.append_message(conv_template.roles[0], text)
    conv_template.append_message(conv_template.roles[1], "")
    prompt = conv_template.get_prompt()
    params = {
        "prompt": {
            "text": prompt,
            "image": encoded_image,
        },
        **generate_kwargs
    }
    timeout = 100
    while True:
        try:
            # starlette StreamingResponse
            response = requests.post(
                worker_addr + "/worker_generate",
                json=params,
                stream=True,
                timeout=timeout,
            )
            if response.status_code == 200:
                worker_initiated = True
            break
        except requests.exceptions.ConnectionError as e:
            if not worker_initiated:
                print("Worker not initiated, waiting for 5 seconds...")
            else:                
                print("Connection error, retrying...")
            time.sleep(5)
        except requests.exceptions.ReadTimeout as e:
            print("Read timeout, adding 10 seconds to timeout and retrying...")
            timeout += 10
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print("Unknown request exception: ", e, "retrying...")
            time.sleep(5)
    try:
        generated_text = json.loads(response.content.decode("utf-8"))['text']
        generated_text = generated_text.strip("\n ")
        # print("Generated text: ", generated_text)
        ic(generated_text)
        return generated_text
    except Exception as e:
        print("Error in worker response: ", e)
        raise e
        return ""
    
def launch_lcoal_worker(
    model_name: str,
) -> str:
    """
    Launch a model worker and return the address
    Args:
        model_name: the model name to launch
    Returns:
        the address of the launched model
    """
    # python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b --port 31011 --worker http://127.0.0.1:31011 --host=127.0.0.1 --no-register
    port = random.randint(30000, 40000)
    worker_addr = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([
        "python3", "-m", "arena.serve.model_worker",
        "--model-path", model_name,
        "--port", str(port),
        "--worker", worker_addr,
        "--host", "127.0.0.1",
        "--no-register",
    ])
    workers.append(proc)
    print(f"Launched model {model_name} at address {worker_addr}")
    return f"http://127.0.0.1:{port}"


def cleanup_process(proc):
    try:
        proc.terminate()  # Try to terminate the process gracefully
        proc.wait(timeout=5)  # Wait for the process to terminate
    except subprocess.TimeoutExpired:
        proc.kill()  # Force kill if not terminated within timeout
    print("Subprocess terminated.")

def main(
    dataset_path: str="WildVision/wildvision-bench",
    dataset_name: str="default",
    dataset_split: str="test",
    worker_addr: str=None,
    model_name: str=None,
    results_dir: str=None,
    LOG_DIR="./logs",
    bench_name="vision_bench_0617"
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
    atexit.register(lambda: [cleanup_process(proc) for proc in workers])
    
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
    
    if any([x in model_name for x in ['gpt', 'gemini', 'claude', 'yi-vl-plus', 'Reka-Flash', 'idefics2-8b-chatty']]):
        # api model
        call_model_worker = partial(call_api_worker_gpt, model_name=model_name, **config)
    else:
        # local model
        if worker_addr is None:
            worker_addr = launch_lcoal_worker(model_name)
        call_model_worker = partial(call_local_worker, worker_addr=worker_addr, model_name=model_name, **config)
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
    
    def map_for_save_to_json(item):
        return {
            "question_id": item['question_id'],
            "instruction": item['instruction'],
            "output": item.get('output', None),
            "token_len": item.get('token_len', None),
            "model": model_name,
            "image": None,
            "language": item['language'],
            "domain": item.get('domain', None),
        }
    images = dataset['image']
    dataset = dataset.remove_columns(["image"])
    dataset = dataset.map(map_for_save_to_json)
    
    
    def map_generate(item, idx): 
        text = item['instruction']
        image = images[idx]
        if 'output' in item and item['output'] is not None:
            pass
        else:
            generated_text = call_model_worker(text, image)
            item['output'] = generated_text
            # FIXME: ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'. If you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={'<|endoftext|>', ...}`. If you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {'<|endoftext|>'})`.
            try:
                item['token_len'] = len(encoding.encode(generated_text))
            except:
                item['token_len'] = len(generated_text.split())
        
        return item
    
    print("Generating...")
    generated_dataset = dataset.map(map_generate, with_indices=True)
    print("Finished generating for all items")
    
    results_file = results_dir + model_name_to_id(model_name) + ".jsonl"
    generated_dataset.to_json(results_file, orient="records", lines=True)
    # new_dataset.save_to_disk(os.path.join(results_dir, model_name))
    print(f"Saved {model_name} answers to {results_file}")
    
    for worker in workers:
        worker.terminate()
    print("Done")


if __name__ == "__main__":
    fire.Fire(main)
