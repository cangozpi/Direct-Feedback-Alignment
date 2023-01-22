import yaml


def read_yaml_config(config_path):
    """
    Inputs:
        config_path (str): path to config.yaml file
    Outputs:
        config (dict): parsed config.yaml parameters
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_tensorboard_hparam_dict(config, lr_schedular):
    """
    Input:
        config (dict): output of read_yaml_config()
    Output:
        hparam_dict (dict): hparam_dict variable to be used with tensorboard SummaryWriter's add_hparams() function
    """
    del config['lr_schedular']
    del config['verbose']
    config['use_lr_schedular'] = lr_schedular['use_lr_schedular']
    config['lr_sched_step_size'] = int(lr_schedular['step_size'])
    config['lr_sched_gamma'] = float(lr_schedular['gamma'])
    return config