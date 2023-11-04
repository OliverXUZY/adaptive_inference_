import torch
import timm
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from libs.model import Worker

def main():
    # print(timm.__version__)
    model_names = timm.list_models('*resnet*50*')
    # pprint(model_names)

    ######################## timm create
    resnet50 = timm.create_model("resnet50.a1_in1k", pretrained=True)
    print("timm.create_model--resnet50.a1_in1k:", len(resnet50.state_dict().items()))


    ######################## timm load from hub
    from timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf
    state_dict = load_state_dict_from_hf("timm/resnet50.a1_in1k")
    print("load_state_dict_from_hf--timm/resnet50.a1_in1k: ", len(state_dict.items()))

    # reorder dict
    # Assuming 'state_dict' is a dictionary containing the model state
    # and 'resnet50.state_dict()' is another dictionary with the desired key order.

    # Create an ordered dictionary with keys in the same order as resnet50.state_dict()
    ordered_state_dict = OrderedDict((key, state_dict[key]) for key in resnet50.state_dict().keys() if key in state_dict)

    com = []
    for (key, val), (mykey, myval) in zip(ordered_state_dict.items(), resnet50.state_dict().items()):
        # Check if keys and values are the same
        # The values are typically tensors, so we use torch.equal to compare them
        com.append((key == mykey) and torch.equal(val, myval))

    # Sum the results to get the number of matches
    total_matches = sum(com)

    # If total_matches is equal to the length of the state_dict (or myresnet.state_dict()),
    # it means all keys and values match.
    print("total_matches for timm_create and hub_load: ", total_matches)

    ######################## customized model
    model_cfg = {'resnet': {'arch': 'resnet50', 'dataset': 'imagenet'},
            'branch_enc': {'embd_dim': 256,
            'out_dim': 128,
            'n_heads': 4,
            'n_layers': 5,
            'attn_pdrop': 0.1,
            'proj_pdrop': 0.1,
            'path_pdrop': 0.1,
            'eos': False,
            'embd_type': 0,
            'pe_type': 0,
            'seq_len': 15},
            'content_enc': {'out_dim': 128,
            'arch': 'resnet10_imagenet',
            'pretrained': False},
            'branch_vae': {'hid_dim': 64, 'n_layers': 3, 'latent_dim': 4, 'in_dim': 15}
            }
    worker = Worker(model_cfg)
    myresnet = worker.resnet

    com = []
    for (key,val),(mykey,myval) in zip(ordered_state_dict.items(), myresnet.state_dict().items()):
        com.append(key == mykey)
    print("total key matches for customized and hub_load: ", sum(com))


if __name__ == "__main__":
    main()