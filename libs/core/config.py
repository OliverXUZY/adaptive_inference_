import yaml


DEFAULTS = {
    # 'seed': 1234567891,

    'data': {
        'dataset': 'imagenet',
        'downsample': True,

        'batch_size': 16,
        'num_workers': 8,
    },

    'model': {
        'resnet': {
            'arch': 'resnet50',
        },

        'branch_enc': {
            'embd_dim': 256,
            'out_dim': 128,
            'n_heads': 4,
            'n_layers': 5,
            'attn_pdrop': 0.1,
            'proj_pdrop': 0.1,
            'path_pdrop': 0.1,
            'eos': True,
            'embd_type': 0,
            'pe_type': 0,
        },

        'content_enc': {
            'out_dim': 128,
            'arch': 'resnet10_imagenet',
            'pretrained': False,
        },
    },

    'opt': {
        'epochs': 8,
        'warmup_epochs': 2,
        'warmup': ['branch_enc', 'content_enc'],
        
        'branch_enc': {
            'optim_type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 0.05,
            'sched_type': 'cosine',
        },

        'content_enc': {
            'optim_type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 0.05,
            'sched_type': 'cosine',
        },

        'branch_vae': {
            'optim_type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 0.05,
            'sched_type': 'cosine',
        },

        'clip_grad_norm': 1.0,
    },

    'train': {
        'n_branches': 64,
        'k': 1,
        'min_n_positives': 16,
        'temperature': 10,
        
        'delta_rs': 0.5,
        'rank_weight': 1.0,
        'sort_weight': 1.0,
        'bce_loss': 0.0,

        'vae_batch_size': 64,
    },

    'eval': {
        'n_branches': 4096,
        'max_macs': 1,
        'batch_size': 256,
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config['model']['resnet']['dataset'] = config['data']['dataset']
    return config