config = {}

config['input_width'] = 1024
config['input_height'] = 256

config['g_input_nc'] = 3*2
config['g_output_nc'] = 3*2

config['g_nfg'] = 24
config['g_layers'] = 9

config['d_input_nc'] = 12
config['d_nfg'] = 6
config['d_layers'] = 3

config['output_dir'] = './model_output_2'

config['evaluation_snapshots_cnt'] = 5
config['evaluation_recursive_samples'] = 3

config['lambda_L1'] = 75

config['adam_lr'] = 0.001
config['adam_b1'] = 0.9
config['adam_b2'] = 0.999
