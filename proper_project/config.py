
class BaseConfig(object):
    
    def __init__(self):

        self.input_width = 1024
        self.input_height = 256

        self.evaluation_snapshots_cnt = 7
        self.evaluation_recursive_samples = 20
        self.full_simulaiton_samples = 25

        self.g_ngf = 0
        self.g_layersl = 0

        self.d_ngf = 0
        self.d_layers = 0

        self.adam_lr = 0.0002
        self.adam_b1 = 0.9
        self.adam_b2 = 0.999

    def argparse(self, parser):
        pass

    def post_config(self):
        pass

        
class SimpleConfig(BaseConfig):

    def __init__(self):
        BaseConfig.__init__(self)

        self.g_ngf = 64
        self.g_layers = 3

        self.d_ngf = 32
        self.d_layers = 3

class AdvanceConfig(BaseConfig):

    def __init__(self):
        BaseConfig.__init__(self)        

        self.g_ngf = 64
        self.g_layers = 7

        self.d_ngf = 48
        self.d_layers = 3

        

default_config = "simple"
configuration_classes = {
    "simple" : SimpleConfig,
    "complicated" : AdvanceConfig
}
