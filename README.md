# WildVision-Bench

## installation

As part of the wildvision arena environment, please install the following dependencies:
```bash
conda create -n wvbench python==3.9
pip install -e .[model_worker,vision_bench]
```

## WVBench Image-Instruct Pair
You can get image-instruct pair of WVBench-500 to generate your model answers by loading bench data below:
```bash
wildbench_data = load_dataset('WildVision/wildvision-arena-data', 'release_bench_0617', split='test500')
```

## Get judgements
First, go to [`config/judge_config.yaml`](config/judge_config.yaml) and add the models you want to evaluate in the `model_list` field. For example:

```yaml
# Add your model below for evaluation
model_list:
  - liuhaotian/llava-v1.6-vicuna-7b
  - openbmb/MiniCPM-V
  - deepseek-ai/deepseek-vl-7b-chat
  - BAAI/Bunny-v1_0-3B
  - Salesforce/instructblip-vicuna-7b
```

Then, run the following command:
```bash
python get_judgement.py
```

Results will be saved in `data/release_bench_0617/model_judgements/judge_gpt-4o_reference_claude-3-sonnet-20240229/{model_name}.jsonl`

## Show the results
```bash
python show_results.py --first-game-only --judge-name gpt-4o --baseline claude-3-sonnet-20240229 --bench-name release_bench_0617
```

