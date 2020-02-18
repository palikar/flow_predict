#!/bin/bash

MODEL_DIRS="$@"

for model_dir in ${MODEL_DIRS[@]}; do

    SEED=$(grep "seed" "${model_dir}/log.txt" | grep "[0-9]*" -o)
    MASK_VAR=$(grep "mask" "${model_dir}/log.txt" | grep "True\|False" -o)
    PRESSURE_VAR=$(grep "use pressure" "${model_dir}/log.txt" | grep "True\|False" -o)

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

        CHECKPOINT=$(ls ${model_dir}/c_unet_*/ | sort -r | head -n1)

        echo "Evaluating ${model_dir}"

        python train.py --data ./data/generated_data/ --model-type 'c' --model-name "unet" --threads 4 --batch-size 3 --shuffle --seed ${SEED} --test-train-split 0.8 --val-train-split 0.1 --output-dir "${model_dir}_new" --evaluate ${PRESSURE_VAR}  --no-train --model-path "${model_dir}/${CHECKPOINT}"  ${MASK_VAR}
        

    fi
    
done
