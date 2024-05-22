from torchvision.transforms import Normalize

def perChannel_mean_std(x,format='NCHW'):
    if format == 'NCHW': return x.mean(axis=(0,2,3)), x.std(axis= (0,2,3))
    elif format == 'NHWC': return x.mean(axis=(0,1,2)), x.std(axis= (0,1,2))
    print('Missing format !!')
    return -1

def perPixel_mean_std(x):
    return x.mean(axis= 0),x.std(axis= 0)

def perChannel_standerization(x):
    '''
    Depricated
    '''
    mean_,std_ = perChannel_mean_std(x)
    return Normalize(mean_,std_)(x)

def subtract_mean_perPixel(x):
    '''
    Depricated
    '''
    mean_,std_ = perPixel_mean_std(x)
    return x-mean_.unsqueeze(0)