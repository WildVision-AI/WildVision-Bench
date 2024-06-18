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

## Acknowledgment
We thank LMSYS for their great work on https://chat.lmsys.org/. Our code base is adapted from https://github.com/lm-sys/arena-hard-auto.

## Citation
```
@misc{lu2024wildvision,
      title={WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences}, 
      author={Yujie Lu and Dongfu Jiang and Wenhu Chen and William Yang Wang and Yejin Choi and Bill Yuchen Lin},
      year={2024},
      eprint={2406.11069},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```
