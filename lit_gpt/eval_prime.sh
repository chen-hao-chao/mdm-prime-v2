python evaluate_prime.py --tasks social_iqa --model prime --batch_size 32 --model_args use_hf=True,cfg=1.0,temp=0.1,mc_num=1024
python evaluate_prime.py --tasks mc_taco --model prime --batch_size 32 --model_args use_hf=True,cfg=1.0,temp=0.75,mc_num=512
python evaluate_prime.py --tasks boolq --model prime --batch_size 32 --model_args muse_hf=True,cfg=0.8,temp=0.8,mc_num=256
python evaluate_prime.py --tasks sciq --model prime --batch_size 32 --model_args use_hf=True,cfg=0.25,temp=0.5,mc_num=256,chunk="0.0_0.75/1_15"
python evaluate_prime.py --tasks arc_easy --model prime --batch_size 32 --model_args use_hf=True,cfg=0.5,temp=1.0,mc_num=512,chunk="0.0_0.5/1_15"
python evaluate_prime.py --tasks anli --model prime --batch_size 32 --model_args use_hf=True,cfg=0.25,temp=0.25,mc_num=128,chunk="0.0/15"
python evaluate_prime.py --tasks truthfulqa_mc1 --model prime --batch_size 32 --model_args use_hf=True,cfg=0.5,temp=0.1,mc_num=256,chunk="0.0/30"
python evaluate_prime.py --tasks openbookqa --model prime --batch_size 32 --model_args use_hf=True,cfg=0.6,temp=0.2,mc_num=512,chunk="0.0_0.25/1_15"