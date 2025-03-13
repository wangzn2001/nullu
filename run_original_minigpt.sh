CUDA_VISIBLE_DEVICES=3 python scripts/model_response.py  --dataset chair --model_name MiniGPT4 --model_path minigpt4/Llama-2-7b-chat-hf --num_samples 500 --num_beams 1 --max_length 64 --seed 42
CUDA_VISIBLE_DEVICES=3 python scripts/model_response.py  --dataset pope --model_name MiniGPT4 --model_path minigpt4/Llama-2-7b-chat-hf --num_samples 500 --num_beams 1 --max_length 64 --seed 42
CUDA_VISIBLE_DEVICES=3 python scripts/model_response.py  --dataset opope --model_name MiniGPT4 --model_path minigpt4/Llama-2-7b-chat-hf --num_samples 500 --num_beams 1 --max_length 256 --seed 42
