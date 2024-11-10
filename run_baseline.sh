dataset="HCPGender"
batch_size="16"
model="GCNConv"
#hidden="64"
main="main.py"
#python $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 10
python $main --dataset HCPGender --model $model --device 'mps' --batch_size $batch_size  --num_layers 3  --epochs 100 --lr 1e-5 --hidden 1000