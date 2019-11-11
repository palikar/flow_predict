

SERVER=arnaudov@i82pc25.itec.kit.edu
LOCATION=~/code/flow_pred/


trian:
	@echo "Running the training script"


evaluate:
	@echo "Running the evaluation script"


simple_run:
	python train.py --config config.json --data ./data/generated_data/


pressure_run:
	python train.py --config config.json --data ./data/generated_data/


# deploy:
# 	@echo "Deploying project files to server"
# 	@ssh -t $(SERVER) 'mkdir -p $(LOCATION)/data'
# 	scp  ./requirements.txt ./config.py ./dataloader.py ./evaluate.py ./models.py ./train.py ./utils.py ./verdana.ttf "$(SERVER):$(LOCATION)/"
# 	scp -r  ./data/Makefile ./data/anim.py ./data/config.xml ./data/cropper.py ./data/gen_data.sh ./data/gen_dataconf_files.py ./data/gen_param_file.py ./data/render_images.sh ./data/geometry_data "$(SERVER):$(LOCATION)/data/"


deploy_training:
	@echo "Deploying the files for the training of models"
	@ssh -t $(SERVER) 'mkdir -p $(LOCATION)/data'
	scp  ./requirements.txt ./dataloader.py ./evaluate.py ./models.py ./train.py ./utils.py ./verdana.ttf "$(SERVER):$(LOCATION)/"

deploy_config:
	@echo "Deploying the config file for the networks"
	@ssh -t $(SERVER) 'mkdir -p $(LOCATION)/data'
	scp  ./config.py


server-train:
	@ssh -t $(SERVER) "cd $(LOCATION)/; python train.py --data ./data/generated_data/ --model-type 'c' --test-train-split 0.8 --evaluate --cuda --model-name 'c_res_l3_nfg64' --use-pressure --threads --batch-size 30 --shuffle --epochs 100 --lr_policy step --print-summeries --val-train-split 0.1"


clean:
	rm -rf ./checkpoints
	rm -rf ./log.txt
	rm -rf ./__pycache__/
