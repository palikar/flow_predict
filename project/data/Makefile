

DATA_DIR=./generated_data
GEN_PARAMS_SCRIPT=./gen_param_file.py
PARAMS_FILE=params.txt
CONFIG_FILE=config.xml
GEOMETRY_DIR=./geometry_data/


MPI_NP=4
DATA_GEN_SCRIPT=./gen_data.sh


run_simulations: gen_params
	@echo "Running simulations"
	@bash $(DATA_GEN_SCRIPT) $(PARAMS_FILE) $(CONFIG_FILE) $(GEOMETRY_DIR) $(DATA_DIR)


gen_params:
	@echo "Generating parameters"
	@python $(GEN_PARAMS_SCRIPT) $(PARAMS_FILE)

clean:
	@echo "Deleting data"
	rm -rf $(DATA_DIR)
	rm -rf $(PARAMS_FILE)
