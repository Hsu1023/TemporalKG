gpu="5"

python run.py -search -earlyStop -space full -dataset ICEWS14 -cpu 2 -valid_steps 1000 -max_steps 50000 -model DEDistMult -eval_test -test_batch_size 16 -gpu $gpu
