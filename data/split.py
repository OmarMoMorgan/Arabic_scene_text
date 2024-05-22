from sklearn.model_selection import train_test_split
from torch  import is_tensor
import numpy as np 

def convert_to_numpy(x):
    if is_tensor(x):
        if x.device != 'cpu':
            x = x.to('cpu')
        x= x.detach().numpy()

    if not isinstance(x, np.ndarray): x= np.array(x)
    return x

def train_test_split_(x,y,config={},stratify = True):
    x_ = convert_to_numpy(x)
    y_ = convert_to_numpy(y)
    if stratify: config['stratify'] = y_
    return train_test_split(x_,y_,**config)




