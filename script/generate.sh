ckdir="gpn_finetune_2datasets_0.8_e5_1"
ckpt="checkpoint-10899"
python generate_embeddings.py \
    --input_csv data/test.csv \
    --checkpoint_dir "output/$ckdir/joint/$ckpt" \
    --batch_size 128 \
    --output_csv "output/$ckdir/joint/$ckpt/$ckdir.csv"