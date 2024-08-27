# WildVision-Bench

## installation

As part of the wildvision arena environment, please install the following dependencies:
```bash
conda create -n visionbench python==3.9
pip install -e .
```

## WVBench Image-Instruct Pair
You can get image-instruct pair of WVBench-500 to generate your model answers by loading bench data below:
```bash
wildbench_data = load_dataset('WildVision/wildvision-bench', split='test')
```

## Generate model answers
```bash
CUDA_VISIBLE_DEVICES=0 python -m gen_answers --model_name "{model_name}" 
# e.g., CUDA_VISIBLE_DEVICES=0 python -m gen_answers --model_name "liuhaotian/llava-v1.6-vicuna-7b"
```

see [`gen_answers.sh`](gen_answers.sh) for more examples.

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

## TODO
- [ ] change the bench to the 0630 version (current 0617)

## Acknowledgment
We thank LMSYS for their great work on https://chat.lmsys.org/. Our code base is adapted from https://github.com/lm-sys/arena-hard-auto.

## Citation
```
@article{lu2024wildvision,
  title={WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences},
  author={Lu, Yujie and Jiang, Dongfu and Chen, Wenhu and Wang, William Yang and Choi, Yejin and Lin, Bill Yuchen},
  journal={arXiv preprint arXiv:2406.11069},
  year={2024}
}
```
