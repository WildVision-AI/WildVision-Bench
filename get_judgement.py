import fire
import yaml
import json
import regex as re
import os
import imagehash
import concurrent.futures
from datasets import load_dataset
from tqdm import tqdm

from bench_utils import (
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    get_endpoint,
    encode_image,
    hash_pil_image,
    load_model_answers,
    load_model_judgements,
    model_name_to_id
)

system_prompt = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""

def get_score(judgement, pattern, pairwise=True):
    matches = pattern.findall(judgement)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False

def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    return output

def judgement(**args):
    question = args["question"]
    images = args["images"] # current assume only one image
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id":question["question_id"],
        "model":answer["model"],
        "judge": model,
        "games":[]
        }

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}
            
            question_1 = question['instruction']
            if game == 0:
                answer1 = baseline['output']
                answer2 = answer['output']
            else:
                answer1 = answer['output']
                answer2 = baseline['output']
            prompt_args["question_1"] = question_1
            prompt_args["answer_1"] = answer1
            prompt_args["answer_2"] = answer2
            
            user_prompt = template.format(**prompt_args)
            user_prompt_with_image = [{"type": "text", "text": user_prompt,}]
            user_prompt_with_image += [
                {
                    "type": "image", "image": image,
                } for image in images
            ]
            
            conv.append({"role": "user", "content": user_prompt_with_image})

        judgement = ""
        for _ in range(2):
            new_judgement = get_answer(
                model,
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
            )

            judgement += ("\n" + new_judgement)

            score, try_again = get_score(judgement, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgement})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgement and finish by outputting a final verdict label"})

        user_prompt = []
        for content in conv[1]["content"]:
            if content["type"] == "text":
                user_prompt.append({"type": "text", "text": content["text"]})
            else:
                user_prompt.append({"type": "image", "image": hash_pil_image(content["image"])})
        result = {
            "user_prompt": user_prompt,
            "judgement": judgement,
            "score":score
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")
        
def main(
    judge_config="./config/judge_config.yaml",
    api_config="./config/api_config.yaml",
):
    with open(judge_config, "r") as f:
        configs = yaml.safe_load(f)
    with open(api_config, "r") as f:
        endpoint_list = yaml.safe_load(f)
    

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    answer_dir = os.path.join("data", configs["bench_name"], "model_answers")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    # load bench data
    questions = load_dataset('WildVision/wildvision-arena-data', 'release_bench_0617_with_modelresponse', split='test500')
    
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model_name_to_id(model) for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgements/judge_{configs['judge_model']}_reference_{configs['baseline_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgements = load_model_judgements(output_dir)
    
    endpoint_info = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                kwargs["images"] = [question["image"]]
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgements and question_id in existing_judgements[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:
                    kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                future = executor.submit(judgement, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgements")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
            
if __name__ == "__main__":
    fire.Fire(main)