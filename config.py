import os
import configparser


class Config:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # experiment
        self.seed = int(parser.get('experiment', 'seed'))

        # training
        self.dataset_path = parser.get('training', 'dataset_path')
        self.save_dir = parser.get('training', 'save_dir')
        self.stage = int(parser.get('training', 'stage'))
        self.log_dir = parser.get('training', 'log_dir')
        self.log_dir = os.path.join(self.save_dir, f'stage{self.stage}_{self.log_dir}')

        self.nThreads = int(parser.get("training", "nThreads"))
        self.num_epochs = int(parser.get("training", "num_epochs"))
        self.lr = float(parser.get("training", "lr"))
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.patch_size = int(parser.get('training', 'patch_size'))
        self.finetuning = (parser.get('training', 'finetuning') == 'True')
        self.save_train_img = (parser.get('training', 'save_train_img') == 'True')

        self.scale = int(parser.get('training', 'scale'))
        self.num_seq = int(parser.get('training', 'num_seq'))

        self.lr_warping_loss_weight = float(parser.get("training", "lr_warping_loss_weight"))
        self.hr_warping_loss_weight = float(parser.get("training", "hr_warping_loss_weight"))
        self.flow_loss_weight = float(parser.get("training", "flow_loss_weight"))
        self.D_TA_loss_weight = float(parser.get("training", "D_TA_loss_weight"))
        self.R_TA_loss_weight = float(parser.get("training", "R_TA_loss_weight"))
        self.Net_D_weight = float(parser.get("training", "Net_D_weight"))

        self.gpu = parser.get("training", "gpu")

        # Network
        self.in_channels = int(parser.get('network', 'in_channels'))
        self.dim = int(parser.get('network', 'dim'))
        self.ds_kernel_size = int(parser.get('network', 'ds_kernel_size'))
        self.us_kernel_size = int(parser.get('network', 'us_kernel_size'))
        self.num_RDB = int(parser.get('network', 'num_RDB'))
        self.growth_rate = int(parser.get('network', 'growth_rate'))
        self.num_dense_layer = int(parser.get('network', 'num_dense_layer'))
        self.num_flow = int(parser.get('network', 'num_flow'))
        self.num_FRMA = int(parser.get('network', 'num_FRMA'))
        self.num_transformer_block = int(parser.get('network', 'num_transformer_block'))
        self.num_heads = int(parser.get('network', 'num_heads'))
        self.LayerNorm_type = parser.get('network', 'LayerNorm_type')
        self.ffn_expansion_factor = float(parser.get('network', 'ffn_expansion_factor'))
        self.bias = (parser.get('network', 'bias') == 'True')

        # validation
        self.val_period = int(parser.get('validation', 'val_period'))

        # test
        self.custom_path = parser.get('test', 'custom_path')
