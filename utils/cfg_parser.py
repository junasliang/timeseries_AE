import yaml

def load_cfg_list(cfg_path='./dry_run.yaml'):
    with open(cfg_path,'r') as file:
        configs = yaml.safe_load(file)
    return configs #output a list of dictionary(ex:{'experiments': None, 'special_test_name': 'experiment_1',...})

def parse_cfg(config:dict):
    print(config, flush=True)
    #parse config
    config_name = config["config_name"]
    min_seq_len = config["min_seq_len"]
    max_seq_len = config["max_seq_len"]
    val_ratio = config["val_ratio"]
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    epochs = config["epoch"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    teacher_forcing = config["teacher_forcing"]
    num_layers = config["layers"]
    return config_name, min_seq_len, max_seq_len, val_ratio, batch_size, hidden_size, epochs, learning_rate, weight_decay, teacher_forcing,num_layers

if __name__=="__main__":
    cfgs = load_cfg_list('./cfg.yaml')
    for cfg in cfgs:
        parse_cfg(cfg)
