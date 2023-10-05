import matlab
import matlab.engine               # import matlab引擎
import matlab
import matlab.engine
import cv2
import numpy as np
import os.path
import copy
import scipy.io

import matplotlib.pyplot as plt

img_path=r'/home/liuqg/wgj/diffu2/lzdata/wang0220/xueguantest60.png'
d = cv2.imread(img_path,0)
d = d / 255.
img=d
img= img.astype(np.float32)
img=img.tolist()
img=matlab.double(img)
engine = matlab.engine.start_matlab()  # 启动matlab engine
sensor_data111=engine.forward2(img)

##
sensor_data111=np.array(sensor_data111)
plt.imshow(sensor_data111)
plt.savefig('aaa1.png')
plt.show()



mask=np.zeros((512,2075))
for i in range(64):
    mask[8*i,:]=1
print(mask)

data=scipy.io.loadmat('512xueguansignal.mat')
yy=data['y']
yy=yy*mask


plt.imshow(yy)
plt.savefig('aaa2.png')
plt.show()



aa=np.zeros((64,2075))
for j in range(64):
    aa[j,:]=yy[8*j,:]


plt.imshow(aa)
plt.savefig('aaa3.png')
plt.show()

#yy=yy.tolist()
#yy=matlab.double(yy)

aa=aa.tolist()
aa=matlab.double(aa)

recon=engine.backward2(aa)
recon=np.array(recon)
print(recon.shape)
print(recon)
recon=(recon-recon.min())/(recon.max()-recon.min())
#cv2.imshow('image', recon) 
cv2.imwrite('./wwresult/512xueguangsignalto64_2.png',255.*recon)
