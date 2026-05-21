SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_name=SeesawNet
model_id_name=SeesawNet_TFMAE_1211
gpu_set=1
itr=3
loss=TFMAE

sh "$SCRIPT_DIR/run/etth1.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/etth2.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/ettm1.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/ettm2.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/exchange_rate.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/weather.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/illness.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/solar.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/electricity.sh" $model_name $model_id_name $gpu_set $itr

