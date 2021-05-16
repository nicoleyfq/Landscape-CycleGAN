import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def dodge(a, b, alpha):
    return min(int(a*255/(256-b*alpha)), 255)


def draw(img, blur=25, alpha=1.0):
    # turn picture to grey
    img1 = img.convert('L')
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)
    # blur the picture
    for i in range(blur):
        img2 = img2.filter(ImageFilter.BLUR)
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            img1.putpixel((x, y), dodge(a, b, alpha))
    

def dodgeNaive(image, mask):
    # determine the shape of the input image
    width, height = image.shape[:2]
 
    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)
 
    for col in range(width):
        for row in range(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)
                # print('tmp={}'.format(tmp.shape))
                # make sure resulting value stays within bounds
                if tmp.any() > 255:
                    tmp = 255
                    blend[col, row] = tmp
 
    return blend
 
 
def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)
 
 
def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)
 
 
def rgb_to_sketch(src_image_name, dst_image_name):
    '''
    this function convert rgb images to sketch style
    '''
    img_rgb = cv2.imread(src_image_name)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #print('read in done')
    # read image and process
    # img_gray = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)
 
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    #print('Guassian Blur done')
    img_blend = dodgeV2(img_gray, img_blur)
 
#     cv2.imshow('original', img_rgb)
#     cv2.imshow('gray', img_gray)
#     cv2.imshow('gray_inv', img_gray_inv)
#     cv2.imshow('gray_blur', img_blur)
#     cv2.imshow("pencil sketch", img_blend)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print('destroy All Windows done')
    cv2.imwrite(dst_image_name, img_blend)
 
def getTrain(train_abs_path):
    '''get the name of train pictures here '''
    train_scenes_list = []
    for (root, dirs,files) in os.walk(train_abs_path):
        for filename in files:
            if filename.endswith('jpg'):
                train_scenes_list.append(filename)

    return train_scenes_list

def main():
    #Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default='Harvard', type=str, required=False,
                        help="The input data dir. Should be Haravd etc.")
    parser.add_argument("--output_dir", default='./data/sketch', type=str, required=False,
                        help="The output data dir.")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #print(args.output_dir)
    data_path = args.subset + '/jpg/'
    img_path = os.path.join('./data/paintings/',data_path)
    save_path= os.path.join(args.output_dir,args.subset)
    #print(save_path)
    data_list = getTrain(img_path)
    print('*****Running sketch process for dataset:',args.subset,'*****')
    print('Size of input dataset is:',len(data_list))
    for i in tqdm(range(len(data_list))):  
        src_image_name = img_path+data_list[i]
        #print(src_image_name)
        dst_image_name = save_path+'/'+data_list[i]
        #print(dst_image_name)    
        rgb_to_sketch(src_image_name, dst_image_name)
    print('*****Finish processing all the images in dataset!*****')

if __name__ == '__main__':
    main()

