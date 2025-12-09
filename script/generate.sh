ckdir="0.8"
ckpt="checkpoint-10755"
python generate_embeddings.py \
    --input_csv data/test.csv \
    --batch_size 96 \
    --checkpoint_dir "output" \
    --output_csv "output/$ckdir.csv"
    # --output_csv "output/$ckdir/joint/$ckpt/$ckdir.csv"
    # --checkpoint_dir "output/$ckdir/joint/$ckpt" \
    # --use_vanilla_gpn \
    # --output_csv "output/vanila.csv" 