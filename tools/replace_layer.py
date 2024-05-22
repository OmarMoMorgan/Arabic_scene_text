import torch
from torch import nn



def replace_fc_layers(module, in_features, out_features,NewModule):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, NewModule(in_features, out_features))
        else:
            replace_fc_layers(child, in_features, out_features)


def replace_specific_layers(module, layer_names, in_features, out_features,NewModule):
    for name, child in module.named_children():
        if name in layer_names and isinstance(child, nn.Linear):
            setattr(module, name, NewModule(in_features, out_features))
        else:
            replace_specific_layers(child, layer_names, in_features, out_features,NewModule)