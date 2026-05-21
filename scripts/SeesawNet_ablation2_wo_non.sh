SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

model_name=SeesawNet_wo_non
model_id_name=SeesawNet_ablation2_wo_non
gpu_set=0
itr=1

sh "$SCRIPT_DIR/run/etth1.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/etth2.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/ettm1.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/ettm2.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/exchange_rate.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/weather.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/illness.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/solar.sh" $model_name $model_id_name $gpu_set $itr
sh "$SCRIPT_DIR/run/electricity.sh" $model_name $model_id_name $gpu_set $itr

