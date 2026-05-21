SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

default_model_name=SeesawNet
default_model_id_name=SeesawNet
default_gpu=0
default_itr=3
default_loss=TFMAE

model_name=${1:-$default_model_name}
model_id_name=${2:-$default_model_id_name}
gpu_ids=${3:-$default_gpu}
itr=${4:-$default_itr}
loss=${5:-$default_loss}

echo 'gpu:' $gpu_ids
echo 'model name:' $model_name
echo 'model id name:' $model_id_name
echo 'iteration time:' $itr
echo 'loss function:' $loss
export CUDA_VISIBLE_DEVICES=$gpu_ids

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/$model_id_name" ]; then
    mkdir ./logs/LongForecasting/$model_id_name
fi

seq_len=104


root_path=./dataset/illness/
data_path=national_illness.csv
data_type=custom
data_name=illness
for pred_len in 24 36 48 60
do
  python -u run.py \
    --is_training 1 \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_type \
    --root_path $root_path \
    --data_path $data_path \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --patch_len 8 \
    --stride 8 \
    --pd_layers 0 \
    --cr_layers 4 \
    --d_model 512 \
    --d_ff 1024 \
    --n_heads 16 \
    --dropout 0.3 \
    --down_sample_rate 0.8 \
    --batch_size 32 \
    --learning_rate 0.0002 \
    --lradj 'type3' \
    --pct_start 0.1 \
    --train_epochs 100 \
    --patience 5 \
    --alpha 0.2 \
    --loss $loss \
    --itr $itr | tee logs/LongForecasting/$model_id_name/$data_name'_'$seq_len'_'$pred_len.logs
done