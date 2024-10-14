python run_vllm.py --tokenizer_mode "mistral" --max_model_len 65536 --num_gpu 1 --model_name mistralai/Pixtral-12B-2409
python run_vllm.py --tokenizer_mode "auto" --max_model_len 32768 --num_gpu 2 --model_name "openbmb/MiniCPM-V-2_6"
python run_vllm.py --tokenizer_mode "auto" --max_model_len 65536 --num_gpu 1 --model_name "Qwen/Qwen2-VL-7B-Instruct"
python run_vllm.py --tokenizer_mode "auto" --max_model_len 65536 --num_gpu 2 --model_name "meta-llama/Llama-3.2-11B-Vision-Instruct"
python run_vllm.py --tokenizer_mode "auto" --max_model_len 65536 --num_gpu 1 --model_name "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
