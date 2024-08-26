import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import PIL.Image as Image
import cv2
import albumentations

RESAMPLE_MODE = Image.BICUBIC
FILL_COLOR = (128, 128, 128)

def GaussianBlur(img, _):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = albumentations.GaussianBlur(always_apply=True, p=1)(image=img)['image']
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def ColorJitter(img, _):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = albumentations.ColorJitter(always_apply=True, p=1)(image=img)['image']
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def ElasticTransform(img, _):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = albumentations.ElasticTransform(always_apply=True, p=1)(image=img)['image']
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def CLAHE(img,_):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = albumentations.CLAHE(always_apply=True, p=1)(image=img)['image']
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Identity(img, _):
    return img

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v, fillcolor=FILL_COLOR)

def Sharpness(img, v):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = albumentations.Sharpen(always_apply=True, p=1)(image=img)['image']
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return img

def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), fillcolor=FILL_COLOR)

def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), fillcolor=FILL_COLOR)

def Solarize(img, v):
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=FILL_COLOR)

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=FILL_COLOR)

def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.6
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (128, 128, 128)  
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Downsample(img, _):
    w, h = img.size
    img = img.resize((w//2,h//2), resample=Image.BILINEAR)
    return img.resize((w,h), resample=Image.BILINEAR)

def ZoomIn(img, v):
    assert 0 <= v <= 0.5
    w, h = img.size
    crop_w = int(v * w)
    crop_h = int(v * h)
    new_w = w - crop_w
    new_h = h - crop_h
    img = PIL.ImageOps.fit(img, size=(new_w,new_h), bleed=v/2, method=Image.BILINEAR)
    return img.resize((w,h), resample=Image.BILINEAR)

def ZoomOut(img, v):
    assert 0 <= v <= 0.5
    w, h = img.size
    pad_w = int(v * w)
    pad_h = int(v * h)
    img = np.asarray(img)
    img = np.pad(img, [(pad_h//2,pad_h//2), (pad_w//2,pad_w//2), (0,0)], mode='reflect')
    img = PIL.Image.fromarray(img)
    return img.resize((w,h), resample=Image.BILINEAR)

def Flip(img, _):  # not from the paper
    if random.random() > 0.5:
        return PIL.ImageOps.flip(img)
    else:
        return PIL.ImageOps.mirror(img)

def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def fixmatch_augment_pool():
    # FixMatch paper
    # op, min_v, max_v
    augs = [
            (Contrast, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (ShearX, -0.2, 0.2),
            (ShearY, -0.2, 0.2),
            (Rotate, -30, 30),
            (Sharpness, 0.1, 1.9),
            (Brightness, 0.5, 1.5),   # 0.1， 1.9
            (TranslateX, -0.2, 0.2),
            (TranslateY, -0.2, 0.2),
            (Identity, 0, 1),
            (CLAHE, 0, 1),
            (ElasticTransform, 0, 1),
            (Downsample, 0, 1),
            # (ColorJitter, 0, 1),
        ]
    return augs

class DFUAugment(object):
    def __init__(self, n, m=13):
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)   # 选择n个,
        # k一共有14个，随意需要改的就是14个参数
        for op, min_val, max_val in ops:
            # 与randaugment不同，这里是使用的改变程度是随机的
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val)
        cutout_val = random.random()*0.5
        img = Cutout(img, cutout_val)
        return img
    
def example():
    path = '/root/autodl-tmp/DataBase/train/1-infection/301258.jpg'
    img = Image.open(path)
    for i in range(10):
        img_list = []
        augs = fixmatch_augment_pool()

        for (op, min_val, max_val) in augs:
            val = min_val + float(max_val - min_val)*random.random()
            img_i = op(img, val)
            # img_i = Image.fromarray(cv2.cvtColor(img_i,cv2.COLOR_BGR2RGB))
            img_list.append(img_i)
        all = 5
        target = Image.new('RGB', (224*all,224*all))
        for row in range(all):
            for col in range(all):
                if row * all + col < len(img_list):
                    target.paste(img_list[all*row+col], (0 + 224*col, 0 + 224*row)) # 显示0,0 后是0,1
        target.save('./exp'+str(i)+'_301258.jpg', quality=100)

# example()