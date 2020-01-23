#!/bin/bash

MODEL_DIRS="$@"



for model_dir in ${MODEL_DIRS[@]}; do

    SEED=$(grep "seed" "${model_dir}/log.txt" | grep "[0-9]*" -o)
    MASK_VAR=$(grep "mask" "${model_dir}/log.txt" | grep "True\|False" -o)
    PRESSURE_VAR=$(grep "use_pressure" "${model_dir}/log.txt" | grep "True\|False" -o)

    if [ $MASK_VAR == "True" ] ; then
        $MASK_VAR=" "
    else
        $MASK_VAR="--no-mask"
    fi

    
    if [ $PRESSURE_VAR == "True" ] ; then
        $PRESSURE_VAR="--use-pressure"
    else
        $PRESSURE_VAR=" "
    fi

    if [ $SEED ] ; then

        echo "Evaluating ${model_dir}"

        echo python train.py --data ./data/generated_data/ --model-type 'c' --model-name "unet" --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${SEED} --test-train-split 0.8 --val-train-split 0.1 --output-dir "${model_dir}_new" --evaluate ${PRESSURE_VAR}  --no-train --model-path ${model_dir}/c_unet_*/*_epoch_45.pth ${MASK_VAR}
        exit
        
        # fi        

        
    fi
    
done
