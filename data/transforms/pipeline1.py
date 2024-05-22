import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mean,std):

        RandomCrop  = A.Compose([A.RandomCrop(28,28,p=0.5),
                                A.PadIfNeeded(32,32)])
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([

                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ToGray(p=0.5),
                A.ColorJitter(p=0.5)

            ]),
            RandomCrop,
            A.Normalize(mean=mean,\
                    std =std ,\
                    max_pixel_value=1.0),
            ToTensorV2()
        ])

        test_transforms = A.Compose(
                    [A.Normalize(mean= mean,\
                    std =std ,\
                    max_pixel_value=1.0),
                    ToTensorV2()])
        
        return train_transforms,test_transforms