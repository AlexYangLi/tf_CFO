CUDA_VISIBLE_DEVICES=2 \
python main.py \
--data_dir ../../data \
--embedding_dim 50 \
--margin_value 4 \
--batch_size 100000 \
--learning_rate 0.01 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 5 \
--max_epoch 100 \
