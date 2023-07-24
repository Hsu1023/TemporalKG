gpu="7"

models=("TTransE""TNTComplEx""TATransE""TADistMult""DEDistMult""DESimplE")

for model in "${models[@]}"
do
python3 run.py -evaluate -dataset $model -cpu 4 -valid_steps 1000 -max_steps 50000 -model DESimplE -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu $gpu -eval_test
done

# max_steps: 200000
# tricks