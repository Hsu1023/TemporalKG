gpu="2"

models=("TNTComplEx" "TATransE" "TADistMult" "DEDistMult" "DESimplE" "TTransE")

for model in "${models[@]}"
do
python3 run.py -evaluate -dataset ICEWS14 -cpu 4 -valid_steps 1000 -max_steps 100000 -model $model -evaluate_times 1 -test_batch_size 16 -data_path ./dataset/ -gpu $gpu -eval_test -earlyStop
done


# max_steps: 200000
# tricks
