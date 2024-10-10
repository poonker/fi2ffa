from __future__ import print_function

from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import os

def showImg(imgName, img, wsize=(400, 400)):
    cv2.namedWindow(imgName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(imgName, wsize[0], wsize[1])
    cv2.imshow(imgName, img)

def homofilter(I):
    I = np.double(I)
    m, n = I.shape
    rL = 0.5
    rH = 2
    c = 2
    d0 = 20
    I1 = np.log(I + 1)
    FI = np.fft.fft2(I1)
    n1 = np.floor(m / 2)
    n2 = np.floor(n / 2)
    D = np.zeros((m, n))
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((i - n1) ** 2 + (j - n2) ** 2)
            H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] / (d0 ** 2)))) + rL
    I2 = np.fft.ifft2(H * FI)
    I3 = np.real(np.exp(I2) - 1)
    I4 = I3 - np.min(I3)
    I4 = I4 / np.max(I4) * 255
    dstImg = np.uint8(I4)
    return dstImg

def gaborfilter(srcImg):
    dstImg = np.zeros(srcImg.shape[0:2])
    filters = []
    ksize = [5, 7, 9, 11, 13]
    j = 0
    for K in range(len(ksize)):
        for i in range(12):
            theta = i * np.pi / 12 + np.pi / 24
            gaborkernel = cv2.getGaborKernel((ksize[K], ksize[K]), sigma=2 * np.pi, theta=theta, lambd=np.pi / 2,
                                             gamma=0.5)
            gaborkernel /= 1.5 * gaborkernel.sum()
            filters.append(gaborkernel)
    for kernel in filters:
        gaborImg = cv2.filter2D(srcImg, cv2.CV_8U, kernel)
        np.maximum(dstImg, gaborImg, dstImg)
    return np.uint8(dstImg)

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
        np.maximum(accum, fimg, accum)
    return accum
def process_re(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
        np.maximum(accum, fimg, accum)
    return accum

def process_threaded(img, filters, threadn=8):
    accum = np.zeros_like(img)

    def f(kern):
        return cv2.filter2D(img, cv2.CV_8U, kern)

    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum

###    Gabor特征提取
def getGabor(img, filters):
    res = []  # 滤波结果
    for i in range(len(filters)):
        res1 = process(img, filters[i])
        res.append(np.asarray(res1))

    pl.figure(2)
    for temp in range(len(res)):
        pl.subplot(4, 6, temp + 1)
        pl.imshow(res[temp], cmap='gray')
    pl.show()
    return res  # 返回滤波结果,结果为24幅图，按照gabor角度排列

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        # kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern = cv2.getGaborKernel((ksize, ksize), 2 * np.pi, theta, 17.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def print_gabor(filters):
    for i in range(len(filters)):
        showImg(str(i), filters[i])

def reverse_image(img):
    antiImg = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            antiImg[i][j] = 255 - img[i][j]
    return antiImg

def pass_mask(mask, img):
    # qwe = reverse_image(img)
    qwe = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                qwe[i][j] = 0
    # asd = cv2.filter2D(qwe, cv2.CV_8U, mask)
    return qwe

def showKern(filters):
    for i in list(range(16)):
        kern = filters[i]
        kern = kern - np.min(kern)
        kern = kern / np.max(kern) * 255
        kern = np.clip(kern, 0, 255)
        kern = np.uint8(kern)
        plt.suptitle('Gabor matched filter kernel')
        plt.subplot(4,4,i+1), plt.imshow(kern, 'gray'), plt.axis('off'), plt.title('theta=' + str(i) + r'/pi')
    plt.show()

def calcDice(predict_img, groundtruth_img):
    predict = predict_img.copy()
    groundtruth = groundtruth_img.copy()
    predict[predict < 128] = 0
    predict[predict >= 128] = 1
    groundtruth[groundtruth < 128] = 0
    groundtruth[groundtruth >= 128] = 1
    predict_n = 1 - predict
    groundtruth_n = 1 - groundtruth
    TP = np.sum(predict * groundtruth)
    FP = np.sum(predict * groundtruth_n)
    TN = np.sum(predict_n * groundtruth_n)
    FN = np.sum(predict_n * groundtruth)
    # print(TP, FP, TN, FN)
    dice = 2 * np.sum(predict * groundtruth) / (np.sum(predict) + np.sum(groundtruth))
    return dice

def adjust_gamma(imgs, gamma=1.0):
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            new_imgs[i, j] = cv2.LUT(np.array(imgs[i, j], dtype=np.uint8), table)
    return new_imgs

def build_filters2(sigma=1, YLength=10):
    filters = []
    widthOfTheKernel = np.ceil(np.sqrt((6 * np.ceil(sigma) + 1) ** 2 + YLength ** 2))
    if np.mod(widthOfTheKernel, 2) == 0:
        widthOfTheKernel = widthOfTheKernel + 1
    widthOfTheKernel = int(widthOfTheKernel)
    # print(widthOfTheKernel)
    for theta in np.arange(0, np.pi, np.pi / 16):
        # theta = np.pi/4
        matchFilterKernel = np.zeros((widthOfTheKernel, widthOfTheKernel), dtype=np.float32)
        for x in range(widthOfTheKernel):
            for y in range(widthOfTheKernel):
                halfLength = (widthOfTheKernel - 1) / 2
                x_ = (x - halfLength) * np.cos(theta) + (y - halfLength) * np.sin(theta)
                y_ = -(x - halfLength) * np.sin(theta) + (y - halfLength) * np.cos(theta)
                if abs(x_) > 3 * np.ceil(sigma):
                    matchFilterKernel[x][y] = 0
                elif abs(y_) > (YLength - 1) / 2:
                    matchFilterKernel[x][y] = 0
                else:
                    matchFilterKernel[x][y] = -np.exp(-.5 * (x_ / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        m = 0.0
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    m = m + 1
        mean = np.sum(matchFilterKernel) / m
        for i in range(matchFilterKernel.shape[0]):
            for j in range(matchFilterKernel.shape[1]):
                if matchFilterKernel[i][j] < 0:
                    matchFilterKernel[i][j] = matchFilterKernel[i][j] - mean
        filters.append(matchFilterKernel)

    return filters

def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x

def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X)))

def Normalize(data):
    k = np.zeros(data.shape, np.float32)
    # k = np.zeros_like(data)
    # m = np.average(data)
    mx = np.max(data)
    mn = np.min(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            k[i][j] = (float(data[i][j]) - mn) / (mx - mn) * 255
    qwe = np.array(k, np.uint8)
    return qwe

def grayStretch(img, m=60.0/255, e=8.0):
    k = np.zeros(img.shape, np.float32)
    ans = np.zeros(img.shape, np.float32)
    mx = np.max(img)
    mn = np.min(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k[i][j] = (float(img[i][j]) - mn) / (mx - mn)
    eps = 0.01
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans[i][j] = 1 / (1 + (m / (k[i][j] + eps)) ** e) * 255
    ans = np.array(ans, np.uint8)
    return ans

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


def get_transform(resize_or_crop,loadSizeX,loadSizeY,fineSize):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop =='scale':
        osize = [loadSizeX,loadSizeY]

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

transform = get_transform('scale',loadSizeX =512, loadSizeY=512, fineSize=512)

def imread(file_path,c=None):
    if c is None:
        im=cv2.imread(file_path)
    else:
        im=cv2.imread(file_path,c)
    
    if im is None:
        raise 'Can not read image'

    if im.ndim==3 and im.shape[2]==3:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im

def imwrite(file_path,image):
    if image.ndim==3 and image.shape[2]==3:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path,image)   

def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        #border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int)
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int32)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-7
    #threhold = np.mean(gray_img)/2
    _, mask = cv2.threshold(gray_img, max(0,threhold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask

# def get_mask_BZ(img):
#     if img.ndim == 3:
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_img = img
    
#     # 使用Otsu's阈值法计算阈值
#     _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # 反转二值化图像
#     inverted_binary = cv2.bitwise_not(binary_img)
    
#     # 去除图像边缘的影响
#     nn_mask = np.zeros((inverted_binary.shape[0]+2, inverted_binary.shape[1]+2), np.uint8)
#     new_mask = inverted_binary.copy()
    
#     # 使用floodFill去除图像边缘的影响
#     cv2.floodFill(new_mask, nn_mask, (0, 0), 255, cv2.FLOODFILL_MASK_ONLY)
#     cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1, new_mask.shape[0]-1), 255, cv2.FLOODFILL_MASK_ONLY)
#     mask = cv2.bitwise_not(new_mask)
    
#     # 使用形态学操作去噪声
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
#     # 将mask的范围缩小到图像范围内
#     mask = mask[:img.shape[0], :img.shape[1]]
    
#     return mask

def _get_center_radius_by_hough(mask):
    #circles= cv2.HoughCircles((mask*255).astype(np.uint8),cv2.HOUGH_GRADIENT,1,1000,param1=5,param2=5,minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2+1)
    circles= cv2.HoughCircles(mask.astype(np.uint8),cv2.HOUGH_GRADIENT,1,1000,param1=5,param2=5,minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2+1)
    center = circles[0,0,:2]
    radius = circles[0,0,2]
    return center,radius

def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    return center_mask    

def get_mask(img):
    if img.ndim ==3:
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    #g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    # center, radius = _get_center_radius_by_hough(tmp_mask)
    # #resize back
    # center = [center[1]*2,center[0]*2]
    # radius = int(radius*2)
    # s_h = max(0,int(center[0] - radius))
    # s_w = max(0, int(center[1] - radius))
    # bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    #tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    #return tmp_mask,bbox,center,radius
    return tmp_mask

def mask_image(img,mask):
    img[mask<=0,...]=0
    return img

def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)    
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border    

# def preprocess(img):
#     # preprocess images 
#     #   img : origin image
#     # return:
#     #   result_img: preprocessed image 
#     #   mask: mask for preprocessed image
#     #mask,bbox,center,radius=get_mask(img)
#     mask=get_mask(img)
#     r_img=mask_image(img,mask)
#     #r_img,r_border=remove_back_area(r_img,bbox=bbox)
#     #mask,_=remove_back_area(mask,border=r_border)
#     #r_img,sup_border=supplemental_black_area(r_img)
#     #mask,_=supplemental_black_area(mask,border=sup_border)
#     return r_img,(mask*255).astype(np.uint8)

#####################################################################3
def process_fi(img):
    # 原图
    # srcImg = cv2.imread(path + ('%02d' % num) + '_test.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    srcImg = cv2.imread(img, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    # 标定图
    # grountruth = cv2.imread(path + ('%02d' % num) + '_manual1.tif', cv2.IMREAD_GRAYSCALE)
    grayImg = cv2.split(srcImg)[1]

    # 提取掩膜
    # ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
    # mask = cv2.erode(th0, np.ones((7, 7), np.uint8))
    # showImg("mask", mask)
    mask = get_mask(grayImg)
    # 高斯滤波
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    # cv2.imwrite("blurImg.png", blurImg)

    # HE
    heImg = cv2.equalizeHist(blurImg)
    # cv2.imwrite("heImg.png", heImg)

    # CLAHE 光均衡化+对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    claheImg = clahe.apply(blurImg)
    # cv2.imwrite("claheImg.png", claheImg)

    # 同态滤波 光均衡化
    homoImg = homofilter(blurImg)

    preMFImg = adjust_gamma(claheImg, gamma=1.5)

    filters = build_filters2()
    # showKern(filters)
    # gaussMFImg = process(grayImg, filters)
    gaussMFImg = process(preMFImg, filters)
    gaussMFImg_mask = pass_mask(mask, gaussMFImg)
    grayStretchImg = grayStretch(gaussMFImg_mask, m=30.0 / 255, e=8)

    # 二值化
    ret1, th1 = cv2.threshold(grayStretchImg, 10, 255, cv2.THRESH_OTSU)
    predictImg1 = th1.copy()

    return predictImg1
    # print(num)
    # dice = calcDice(predictImg, grountruth)
    # print(num,'',dice)
    # wtf = np.hstack([srcImg, cv2.cvtColor(grountruth,cv2.COLOR_GRAY2BGR),cv2.cvtColor(predictImg,cv2.COLOR_GRAY2BGR)])
    # cv2.imwrite(('m%02d' % num)+'.png', wtf)
    # cv2.imwrite(predictImg)
    # cv2.imshow('predict', predictImg1)
    # cv2.imshow('ori', srcImg)
    # # cv2.imshow('temp', preMFImg)
    # cv2.waitKey(0)
    # # cv2.waitKey()
    # cv2.destroyAllWindows()

###################################################################
def process_ffa(img):
    # 原图
    # srcImg = cv2.imread(path + ('%02d' % num) + '_test.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    #srcImg = cv2.imread('001.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    srcImg = cv2.imread(img,1)
    # 标定图
    # grountruth = cv2.imread(path + ('%02d' % num) + '_manual1.tif', cv2.IMREAD_GRAYSCALE)
    grayImg = cv2.split(srcImg)[1]

    # 提取掩膜
    # ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
    # ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
    # mask = cv2.erode(th0, np.ones((7, 7), np.uint8))
    mask = get_mask(grayImg)
    mask = (mask*255)
    # mask = cv2.bitwise_not(mask)
    # mask = cv2.erode(th0, np.zeros((7, 7), np.uint8))
    #showImg("mask", mask)

    #反转图像像素并进行mask
    grayImg = cv2.bitwise_not(grayImg)
    grayImg=mask_image(grayImg,mask)

    # 高斯滤波
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    # cv2.imwrite("blurImg.png", blurImg)

    # HE
    heImg = cv2.equalizeHist(blurImg)
    # cv2.imwrite("heImg.png", heImg)

    # CLAHE 光均衡化+对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    claheImg = clahe.apply(blurImg)
    # cv2.imwrite("claheImg.png", claheImg)

    # 同态滤波 光均衡化
    homoImg = homofilter(blurImg)

    preMFImg = adjust_gamma(claheImg, gamma=1.5)
    filters = build_filters2()
    # showKern(filters)
    gaussMFImg = process_re(preMFImg, filters)
    gaussMFImg_mask = pass_mask(mask, gaussMFImg)
    grayStretchImg = grayStretch(gaussMFImg_mask, m=30.0 / 255, e=8)



    # 二值化
    ret1, th1 = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_OTSU)
    # ret1, th1 = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_BINARY)
    predictImg = th1.copy()

    # print(num)
    # dice = calcDice(predictImg, grountruth)
    # print(num,'',dice)
    # wtf = np.hstack([srcImg, cv2.cvtColor(grountruth,cv2.COLOR_GRAY2BGR),cv2.cvtColor(predictImg,cv2.COLOR_GRAY2BGR)])
    # cv2.imwrite(('m%02d' % num)+'.png', wtf)
    # cv2.imwrite(predictImg)
    # cv2.imshow('predict', predictImg)
    # cv2.imshow('ori', srcImg)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    # # cv2.waitKey()
    # cv2.destroyAllWindows()
    return predictImg
    #############################################################################
def get_union(img1,img2,save_folder,img_name):
    # img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

    # 对两张图像进行按位与操作
    # result = cv2.bitwise_or(img1, img2)
    result = cv2.bitwise_and(img1, img2)
    save_path = os.path.join(save_folder,img_name)
    cv2.imwrite(save_path,result)

    # # 显示结果图像
    # cv2.imshow('Result', result)
    # cv2.imshow('predictImg1', predictImg1)
    # cv2.imshow('predictImg', predictImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('write in: ',save_path)
    
################################################################
def process_images_from_folders(folder_fi, folder_ffa,folder_save):
    # 获取文件夹中的所有文件名
    images_fi = sorted([f for f in os.listdir(folder_fi) if os.path.isfile(os.path.join(folder_fi, f))])
    images_ffa = sorted([f for f in os.listdir(folder_ffa) if os.path.isfile(os.path.join(folder_ffa, f))])
    # 确保输出文件夹存在
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)    
    # print(images_fi)
    # 处理文件夹1中的图像
    # for img_name in images_fi:
    #     img_path_fi = os.path.join(folder_fi, img_name)
    #     fi_seg = process_fi(img_path_fi)
    #     # cv2.imshow(f'Processed Image from {folder1}: {img_name}', processed_image1)

    # # 处理文件夹2中的图像
    # for img_name in images_ffa:
    #     img_path_ffa = os.path.join(folder_ffa, img_name)
    #     ffa_seg = process_ffa(img_path_ffa)
        # cv2.imshow(f'Processed Image from {folder2}: {img_name}', ffa_seg)

    for img_name in images_fi:
        img_path_fi = os.path.join(folder_fi, img_name)
        fi_seg = process_fi(img_path_fi)
        img_path_ffa = os.path.join(folder_ffa, img_name)
        ffa_seg = process_ffa(img_path_ffa)    
        get_union(fi_seg,ffa_seg,folder_save,img_name)    
        # cv2.imshow(f'Processed Image from {folder1}: {img_name}', processed_image1)

    # # 处理文件夹2中的图像
    # for img_name in images_ffa:
    #     img_path_ffa = os.path.join(folder_ffa, img_name)
    #     ffa_seg = process_ffa(img_path_ffa)

    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 使用示例
folder_fi = '/root/fi_enhanced_registed_433/image'
folder_ffa = '/root/ffa_enhanced_registed_433/image'
folder_save = '/root/ffa_enhanced_registed_433_segand/image'
process_images_from_folders(folder_fi, folder_ffa,folder_save)