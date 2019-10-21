#!/bin/bash

MODEL_FILE=$1
MODEL_TYPE=$2

while IFS=" " read -r model_weights
do
    model_name=$(basename ${model_weights%".pth"})

		output_dir="./eval/eval_${model_name}"

		sed ./config.py -i -e "s#config\['output_dir'\]\s=\s'.*'#config['output_dir'] = '${output_dir}'#g"

		echo python train.py --data ./data/generated_data/ --model-type "${MODEL_TYPE}" --test-train-split 0.8 --cuda --model-name 'c_res_l3_nfg64' --threads --batch-size 30 --shuffle --epochs 100 --lr_policy step --model-path "${model_weights}" --evaluate --no-train
		echo $output_dir

done < ${MODEL_FILE}


