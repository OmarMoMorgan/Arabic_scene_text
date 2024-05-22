from .ResNetA import ResNetA
from .plainNet import plainNet
from .sub_CNN import sub_CNN
from .KAN import KAN

def build_model(model_name,config,groups,num_filters):
    if model_name =='resnet':
        model = ResNetA(config)
    elif model_name == 'plain_net': 
        model = plainNet(config)
    elif model_name == 'sub_CNN':
        model = sub_CNN(config,groups,num_filters=num_filters);
    
    return model
    