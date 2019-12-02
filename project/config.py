config = {}

config['input_width'] = 1024
config['input_height'] = 256

config['g_input_nc'] = 3*2
config['g_output_nc'] = 3

config['g_nfg'] = 34
config['g_layers'] = 5

config['d_input_nc'] = 12

config['d_nfg'] = 38
config['d_layers'] = 6

config['output_dir'] = './model_output'

config['evaluation_snapshots_cnt'] = 7
config['evaluation_recursive_samples'] = 20
config['full_simulaiton_samples'] = 25

config['lambda_L1'] = 100

config['adam_lr'] = 0.0002
config['adam_b1'] = 0.9
config['adam_b2'] = 0.999


# config['input_width'] = 1024
# config['input_height'] = 256

# config['g_input_nc'] = 3
# config['g_output_nc'] = 2

# config['g_nfg'] = 34
# config['g_layers'] = 5

# config['d_input_nc'] = 12

# config['d_nfg'] = 22
# config['d_layers'] = 4

# config['output_dir'] = './model_output'

# config['evaluation_snapshots_cnt'] = 5
# config['evaluation_recursive_samples'] = 5
# config['full_simulaiton_samples'] = 5

# config['lambda_L1'] = 100

# config['adam_lr'] = 0.001
# config['adam_b1'] = 0.9
# config['adam_b2'] = 0.999
