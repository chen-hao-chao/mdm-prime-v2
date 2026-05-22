# seeds and temperatures are choosen to align the entropy (diversity)
python sampling_uncond.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --nfe 128 --seed 0 1 2 3 4
python sampling_uncond.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --nfe 256 --seed 2 3 4 5 6
python sampling_uncond.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --nfe 512 --seed 11 15 19 28 33 --temperature 1.01
python sampling_uncond.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --nfe 768 --seed 0 1 2 3 4
python sampling_uncond.py --model_name='chen-hao-chao/mdm-prime-v2-slimpajama' --nfe 1024 --seed 10 11 12 13 14 --temperature 1.01