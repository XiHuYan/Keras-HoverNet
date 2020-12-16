import albumentations as albu

def center_crop(img, input_shape):
    augumented = albu.CenterCrop(input_shape[0], input_shape[1], always_apply=True)
    augs = augumented(image=img)
    return augs['image']

def shape_aug(img, lbl, input_shape=(270,270)):
    assert img.dtype=='uint8' 
    assert lbl.dtype=='uint8'

    augumented =  albu.Compose([ 
        albu.IAAAffine(scale=(0.8,1.2),
                      rotate=179, 
                      shear=5, 
                      translate_percent=(0.01,0.01),
                      mode='constant', 
                      p=0.5),
            albu.Flip(p=.5),
            albu.CenterCrop(input_shape[0], input_shape[1], always_apply=True)
        ])

    augs = augumented(image=img, mask=lbl)
    img, lbl = augs['image'], augs['mask']
    return img, lbl

def color_aug(img):
    augumented = albu.Compose([   
                    albu.OneOf([
                        albu.GaussianBlur(p=1),
                        albu.MedianBlur(p=1),
                        albu.GaussNoise(p=1),
                    ], p=.5),

                    albu.ColorJitter(brightness=0.2, contrast=((0.75,1.25)), saturation=0.2, hue=(-0.04,0.04), p=0.5) 
                ])

    image = augumented(image=img)['image']
    return image


    
