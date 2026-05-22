# seeds are choosen to align the entropy (diversity)
python sampling_uncond.py --model_name='nieshen/SMDM' --nfe 128 --seed 1 2 3 4 5
python sampling_uncond.py --model_name='nieshen/SMDM' --nfe 256 --seed 8 17 18 19 20
python sampling_uncond.py --model_name='nieshen/SMDM' --nfe 512 --seed 0 1 2 3 4
python sampling_uncond.py --model_name='nieshen/SMDM' --nfe 768 --seed 0 1 2 3 4
python sampling_uncond.py --model_name='nieshen/SMDM' --nfe 1024 --seed 0 1 2 3 4
