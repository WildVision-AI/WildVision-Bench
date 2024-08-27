import fire
import json
import yaml
import tiktoken
from pathlib import Path

avaliable_wvbench_versions = [
    "wildvision_0617"
]
def main(
    lmmseval_log_dir:str,
    model_name:str,
):
    lmmseval_log_dir = Path(lmmseval_log_dir)
    results_path = lmmseval_log_dir / "results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    wvbench_version = list(results["configs"].keys())[0]
    assert wvbench_version in avaliable_wvbench_versions, f"{wvbench_version} not in {avaliable_wvbench_versions}, please check if it's expected"
    generation_config = results["configs"][wvbench_version]['generation_kwargs']
    
    model_config_path = Path(__file__).parent / "model_configs" / model_name / "generation_config.yaml"
    if model_config_path.exists():
        print(f"model config already exists at {model_config_path}, please check if it's expected")
        return
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_config_path, "w") as f:
        yaml.dump(generation_config, f)
    print(f"model config saved at {model_config_path}")
    
    full_results_path = lmmseval_log_dir / f"{wvbench_version}.json"
    
    version_date = wvbench_version.split("_")[1]
    local_wvbench_name = "vision_bench"
    model_answer_save_file = Path(__file__).parent / "data" / f"{local_wvbench_name}_{version_date}" / "model_answers" / f"{model_name}.jsonl"
    if model_answer_save_file.exists():
        print(f"model answers already exists at {model_answer_save_file}, please check if it's expected")
        return

    with open(full_results_path, "r") as f:
        full_results = json.load(f)
    logs = full_results["logs"]
    all_model_answers = []
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for log in logs:
        doc = log["doc"]
        output = log["filtered_resps"][0]
        model_answer = {
            "question_id": doc["question_id"],
            "instruction": doc["instruction"],
            "output": output,
            "model": model_name,
            "language": doc["language"],
            "token_len": len(encoding.encode(output, disallowed_special=()))
        }
        all_model_answers.append(model_answer)

    with open(model_answer_save_file, "w") as f:
        for model_answer in all_model_answers:
            f.write(json.dumps(model_answer) + "\n")
    print(f"model answers saved at {model_answer_save_file}")
    
    judge = full_results['model_configs']['metadata']['judge_model']
    baseline_model = full_results['model_configs']['metadata']['baseline_model']
    model_judgement_save_file = Path(__file__).parent / "data" / f"{local_wvbench_name}_{version_date}" / "model_judgements" / f"judge_{judge}_reference_{baseline_model}" / f"{model_name}.jsonl"
    if model_judgement_save_file.exists():
        print(f"model judgements already exists at {model_judgement_save_file}, please check if it's expected")
        return
    model_judgements = []
    for log in logs:
        doc = log["doc"]
        model_judgement = {
            "question_id": doc["question_id"],
            "model": model_name,
            "judge": judge,
            "games": [
                {
                    "user_prompt": None,
                    "judgement": log['gpt_eval_score']['gpt_resps'],
                    "score": log['gpt_eval_score']['filtered_resps']
                }
            ]
        }
        model_judgements.append(model_judgement)
    with open(model_judgement_save_file, "w") as f: 
        for model_judgement in model_judgements:
            f.write(json.dumps(model_judgement) + "\n")
    print(f"model judgements saved at {model_judgement_save_file}")
    
    
    
if __name__ == "__main__":
    fire.Fire(main)