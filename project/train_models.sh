#!/bin/bash


NGF=32
LAYERS=${LAYERS}

NETS=()

while [ "$1" != "" ]; do
    
    PARAM=`echo $1 | awk -F= '{print $1}'`

    case $PARAM in
        
        --pressure | -p)
            PRESSURE=1
            ;;
        --unet | -u)
            NETS=( "${net}" "${NETS[@]}" )
            ;;
        --res | -r)
            NETS=( "res" "${NETS[@]}" )
            ;;
        --constant | -c)
            CONST=1
            ;;
        --speed | -s)
            SPEED=1
            ;;
        --fluid | -f)
            FLUID=1
            ;;

        --layers | -l)
            LAYERS=`echo $1 | awk -F= '{print $2}'`
            ;;

        --ngf | -n)
            NGF=`echo $1 | awk -F= '{print $2}'`
            ;;
        
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            exit 1
            ;;
    esac
    
    shift
    
done


for net in ${NETS[@]}
do

    ##########################################################
    #  ____  _       _         __  __           _      _     #
    # |  _ \| | __ _(_)_ __   |  \/  | ___   __| | ___| |___ #
    # | |_) | |/ _` | | '_ \  | |\/| |/ _ \ / _` |/ _ \ / __|#
    # |  __/| | (_| | | | | | | |  | | (_) | (_| |  __/ \___ #
    # |_|   |_|\__,_|_|_| |_| |_|  |_|\___/ \__,_|\___|_|___/#
    ##########################################################


    if [ ! -z "$CONST" ]; then

        
        if [ ! -z "$PRESSURE" ]; then
            
            python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_1/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_2/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_3/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_4/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure
            
        fi

        
        python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_5/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_6/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_7/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'c' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_c/plain_results_8/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

    fi

    #######################################################
    #######################################################


    #################################################################
    #  ____                      _   __  __           _      _      #
    # / ___| _ __   ___  ___  __| | |  \/  | ___   __| | ___| |___  #
    # \___ \| '_ \ / _ \/ _ \/ _` | | |\/| |/ _ \ / _` |/ _ \ / __| #
    #  ___) | |_) |  __/  __/ (_| | | |  | | (_) | (_| |  __/ \__   #
    # |____/| .__/ \___|\___|\__,_| |_|  |_|\___/ \__,_|\___|_|___/ #
    #       |_|                                                     #
    #################################################################
    
    if [ ! -z "$SPEED" ]; then

    
        if [ ! -z "$PRESSURE" ]; then
    
            python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_1/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_2/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_3/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_4/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure
    
        fi

    
        python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_5/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_6/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_7/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 's' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_8/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

    fi

    #################################################################
    #################################################################
    #################################################################

    #################################################################
    #  _____ _       _     _   __  __           _      _            #
    # |  ___| |_   _(_) __| | |  \/  | ___   __| | ___| |___        #
    # | |_  | | | | | |/ _` | | |\/| |/ _ \ / _` |/ _ \ / __|       #
    # |  _| | | |_| | | (_| | | |  | | (_) | (_| |  __/ \__         #
    # |_|   |_|\__,_|_|\__,_| |_|  |_|\___/ \__,_|\___|_|___/       #
    #################################################################



    if [ ! -z "$FLUID" ]; then

        
        if [ ! -z "$PRESSURE" ]; then
            
            python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_1/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_2/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_3/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure

            python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_4/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS} --use-pressure
            
        fi
        
        
        python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3 --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_5/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_6/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_7/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        python train.py --data ./data/generated_data/ --model-type 'vd' --cuda --model-name '${net}' --threads 4 --batch-size 3  --shuffle --epochs 50 --lr_policy step --seed ${RANDOM} --print-summeries --test-train-split 0.8 --val-train-split 0.1 --output-dir './results_s/plain_results_8/' --evaluate --g_nfg ${NGF} --g_layers ${LAYERS}

        
    fi


    #################################################################
    #################################################################
    #################################################################

    
    
done

