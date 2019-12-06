#!/bin/bash

MODEL_DIRS="$@"



for model_dir in ${MODEL_DIRS[@]}; do

    SEED=$(grep "seed" "${model_dir}/log.txt" | grep "[0-9]*" -o)

    if [ $SEED ] ; then

        # if [ ! -d ${model_dir}/test ] ; then

        echo "Evaluating ${model_dir}"

        echo python train.py --data ./data/generated_data/ --model-type 'c' --model-name "unet" --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed $SEED --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir "${model_dir}_new" --evaluate --use-pressure  --no-train --model-path ${model_dir}/c_unet_*/* --no-mask
        exit
        
        # fi        

        
    fi
    
done
