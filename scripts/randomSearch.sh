gpu="2"

python run.py -search -earlyStop -space full -dataset ICEWS0515 -cpu 2 -valid_steps 1000 -max_steps 50000 -model TTransE -eval_test -test_batch_size 16 -gpu $gpu
