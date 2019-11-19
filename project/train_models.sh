#!/bin/bash



python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_1/' --evaluate --g_nfg 34 --g_layers 5

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_2/' --evaluate --g_nfg 34 --g_layers 5

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_3/' --evaluate --g_nfg 34 --g_layers 5

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_4/' --evaluate --g_nfg 34 --g_layers 5




python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_5/' --evaluate --g_nfg 34 --g_layers 5 --use-pressure

_python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_6/' --evaluate --g_nfg 34 --g_layers 5 --use-pressure

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_7/' --evaluate --g_nfg 34 --g_layers 5 --use-pressure

python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'unet' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_8/' --evaluate --g_nfg 34 --g_layers 5 --use-pressure



# python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name 'res' --threads 4 --batch-size 12  --shuffle --epochs  80 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results/plain_results_10/' --evaluate --g_nfg 34 --g_layers 5 --use-pressure

