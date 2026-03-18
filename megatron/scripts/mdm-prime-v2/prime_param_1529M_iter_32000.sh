export NLAYERS=24 NHIDDEN=2304 NHEADS=18 FFN_SIZE=9216
export MICRO_BS=16 GBS=512 TRAIN_ITERS=32000 SEQ=2048
export PARAM=1529
export TP_NUM=2
export VARIANT="prime_param_${PARAM}M_iter_${TRAIN_ITERS}"
export GPUS_PER_NODE=8

torchrun --nproc_per_node=$GPUS_PER_NODE pretrain_mask_diff.py \
  --use-mcore-models \
  --num-layers $NLAYERS \
  --hidden-size $NHIDDEN \
  --ffn-hidden-size $FFN_SIZE \
  --num-attention-heads $NHEADS \
  --seq-length $SEQ \
  --max-position-embeddings $SEQ \
  --position-embedding-type rope \
  --use-rotary-position-embeddings \
  --swiglu \
  --normalization RMSNorm \
  --disable-bias-linear \
  --qk-layernorm \
  --embedding-type cat-share \
  --subtokenizer-type baseb_shuffle \
  --random-ratio 1.0 \
  --target-length 16 \
  --padded-output 1 \
  --padded-input 1 \
  --vocab-size 50257 \
  --untie-embeddings-and-output-weights \
  --legacy-tokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file download/tokenizer/gpt2_tok/vocab.json \
  --merge-file download/tokenizer/gpt2_tok/merges.txt \
  --train-data-path download/megatron_c4_en_train_text_document \
  --valid-data-path download/megatron_c4_en_val_text_document \
  --bf16 \
  --use-flash-attn \
  --recompute-activations \
  --overlap-grad-reduce \
  --sequence-parallel \
  --tensor-model-parallel-size $TP_NUM \
  --pipeline-model-parallel-size 1 \
  --num-workers $GPUS_PER_NODE \
  --micro-batch-size $MICRO_BS \
  --global-batch-size $GBS \
  --train-iters $TRAIN_ITERS \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.1 \
  --lr 2e-4 --min-lr 2e-5 \
  --lr-warmup-iters 1000 \
  --lr-decay-style cosine \
  --clip-grad 1.0 \
  --save-interval 2000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --log-interval 10 \
  --ckpt-fully-parallel-save \
  --save checkpoints/$VARIANT \
  --wandb-project megatron_LM \
  --wandb-exp-name $VARIANT \
  --wandb-save-dir outputs/wandb_$VARIANT \
  --tensorboard-dir outputs/tensorboard_$VARIANT \
  --tensorboard-queue-size 5 \
  --log-timers-to-tensorboard \
  --log-validation-ppl-to-tensorboard