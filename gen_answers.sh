CUDA_VISIBLE_DEVICES=0 python -m gen_answers --model_name "THUDM/cogvlm-chat-hf"
# CUDA_VISIBLE_DEVICES=0 python -m gen_answers --model_name "THUDM/cogvlm-chat-hf"
CUDA_VISIBLE_DEVICES=1 python -m gen_answers --model_name "liuhaotian/llava-v1.5-13b" 
CUDA_VISIBLE_DEVICES=1 python -m gen_answers --model_name "unum-cloud/uform-gen2-qwen-500m"
# CUDA_VISIBLE_DEVICES=2 python -m gen_answers --model_name "bczhou/tiny-llava-v1-hf"
CUDA_VISIBLE_DEVICES=1 python -m gen_answers --model_name "deepseek-ai/deepseek-vl-7b-chat"
CUDA_VISIBLE_DEVICES=3 python -m gen_answers --model_name "BAAI/Bunny-v1_0-3B"
CUDA_VISIBLE_DEVICES=3 python -m gen_answers --model_name "Qwen/Qwen-VL-Chat"
CUDA_VISIBLE_DEVICES=0 python -m gen_answers --model_name "openbmb/MiniCPM-V"

CUDA_VISIBLE_DEVICES=0,1 python -m gen_answers --model_name "liuhaotian/llava-v1.6-34b"
# CUDA_VISIBLE_DEVICES=6,7 python -m gen_answers --model_name "liuhaotian/llava-v1.6-34b"

# export OPENAI_API_KEY=sk-xxx
python -m gen_answers --model_name "gpt-4-turbo-2024-04-09"

# export ANTHROPIC_API_KEY="sk-xxx"
python -m gen_answers --model_name "gemini-pro-vision" &

# export GOOGLE_API_KEY="xxx"
# python -m gen_answers --model_name "claude-3-opus-20240229"




  
  - BAAI/Bunny-v1_0-3B
  - bczhou/tiny-llava-v1-hf
  - deepseek-ai/deepseek-vl-7b-chat
  - liuhaotian/llava-v1.5-13b
  - liuhaotian/llava-v1.6-34b
  - liuhaotian/llava-v1.6-vicuna-13b
  - openbmb/MiniCPM-V
  - Salesforce/instructblip-vicuna-7b
  - unum-cloud/uform-gen2-qwen-500m  
  - liuhaotian/llava-v1.6-vicuna-7b
  - THUDM/cogvlm-chat-hf

  - Qwen/Qwen-VL-Chat

# python -m gen_answers --model_name "gpt-4-turbo-2024-04-09"
python -m gen_answers --model_name "gpt-4-vision-preview" & python -m gen_answers --model_name "gpt-4o"

python -m gen_answers --model_name "claude-3-opus-20240229" &
python -m gen_answers --model_name "claude-3-sonnet-20240229" &
python -m gen_answers --model_name "claude-3-haiku-20240307"

# for http image on Dongfu's darth server.
export WEB_IMG_FOLDER="./http_img"
export WEB_IMG_URL_ROOT="https://tigerai.ca/http_img"

python -m gen_answers --model_name "gemini-pro-vision"

python -m gen_answers --model_name "yi-vl-plus" &
python -m gen_answers --model_name "Reka-Flash" &
python -m gen_answers --model_name "idefics2-8b-chatty" &