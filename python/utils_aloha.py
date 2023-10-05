# -*- ecoding: utf-8 -*-
# @ModuleName: col_sub2ind
# @Function:
# @Author: MieMie
# @Time: 2022/5/31 16:40
# @Info:
import numpy as np
from lmafit_mc_adp_v2 import lmafit_mc_adp
import scipy.io as io
import torch
def im2row(im,winSize):
  size = (im).shape
  out = np.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=np.float32)
  count = -1
  for y in range(winSize[1]):
    for x in range(winSize[0]):
      count = count + 1                 
      temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
      temp2 = np.reshape(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')
      out[:,count,:] = np.squeeze(temp2) # MATLAB reshape          
		
  return out
  
def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape #(63001, 36, 8)
    sx = size_data[0] # 256
    sy = size_data[1] # 256
    sz = size_mtx[2] # 8
    
    res = np.zeros((sx,sy,sz),dtype=np.float32)
    W = np.zeros((sx,sy,sz),dtype=np.float32)
    out = np.zeros((sx,sy,sz),dtype=np.float32)
    count = -1
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + np.reshape(np.squeeze(mtx[:,count,:]),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    out = np.multiply(res,1./W)
    return out





def col_ind2sub(array_shape, ind):

    ind[ind < 0] = -1

    ind[ind >= array_shape[0]*array_shape[1]] = -1

    rows = (ind.astype('int') / array_shape[1])

    cols = ind % array_shape[1]

    return (rows, cols)



def find_num(Z,arr):
    
    list1 = Z.flatten(order='F')
    list2 = []
    
    for i in range(len(arr)):
        
        list2.append(list1[arr[i]])


    return np.array(list2)[:,np.newaxis]
    
    
def find_index(Z, num=None):
    
    # num存在时，找num对应的数 
    # 不存在时，找不为0的数
    list1 = Z.flatten(order='F')
    list2 = []
    
    if num:
        
        for i in range(len(list1)):
            if list1[i] == num:
                list2.append(i)
    else:
       
        for i in range(len(list1)):
            if list1[i] != 0:
                list2.append(i)
    
    return list2
    
# a = np.array([[1,3, 1, 4, 5], [2, 6, 1, 2, 1], [3, 7 ,5 ,3, 8],[4, 4, 1, 3, 6], [5, 2, 2, 7, 9]] )
# img = np.pad(a,((1,1),(1,1)),'constant')
# imgp = np.zeros((7,7,3),dtype=np.float32)

# imgp[:,:,0]=img
# imgp[:,:,1]=img
# imgp[:,:,2]=img




# mask1 = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0 ,0], [1 ,0, 1, 0, 0],[1, 0 ,0, 1, 0], [0, 0, 0, 1, 1 ]] )
# # mask=repmat(mask1,[1 1 3]);
# mask = np.pad(mask1,((1,1),(1,1)),'constant')

# maskp = np.zeros((7,7,3),dtype=np.float32)

# maskp[:,:,0]=mask;
# maskp[:,:,1]=mask;
# maskp[:,:,2]=mask;

# imgp = imgp/imgp.max();


# dimg=imgp*maskp;
# Nimg=3;
# Nfir=2;  

Nimg=45;
Nfir=13; 

# maskp=padarray(mask,[hNimg,hNimg]);

# rmask       = maskp[1:4,1:4,:]
rmask       = io.loadmat('./rmask.mat')['rmask']

mask_cmtx   = im2row(rmask,[Nfir,Nfir])
size_temp = mask_cmtx.shape
mask_cmtx = np.reshape(mask_cmtx,[size_temp[0],size_temp[1]*size_temp[2]],order = 'F')  # (62001, 192)



# rval        = dimg[1:4,1:4,:]

rval       = io.loadmat('./rval.mat')['rval']

cmtx        = im2row(rval,[Nfir,Nfir])
size_temp1 = cmtx.shape
cmtx = np.reshape(cmtx,[size_temp1[0],size_temp1[1]*size_temp1[2]],order = 'F')


M1 = mask_cmtx
M1 = np.reshape(M1.T,[size_temp[0]*size_temp[1]*size_temp[2],1])
Known = np.where(M1==1)[0]  #  (100017,)

# Known = np.array(Known)

M2 = np.reshape(cmtx.T,[size_temp[0]*size_temp[1]*size_temp[2],1])
data = M2[Known]  # (100017, 1)

# print(np.round(data[:,0],3))
# print(type(data[0]))

# print(data.dtype)
# assert 0

#data1=io.loadmat(r'.\data.mat')['data'].astype(np.float32)  # (100017, 1)  # float64
#Known1=io.loadmat(r'.\Known.mat')['Known'][:,0]  # (100017, )

# print(data1.dtype)
# assert 0
# print(np.round(data1[:,0],3))


# temp = np.round(data[:,0],3) - np.round(data1[:,0],3)
# print(temp.shape)
#print(type(data),data.shape,data,data.dtype)
#print(type(data1),data1.shape,data1,data1.dtype)
#print(type(Known),Known.shape,Known,Known.dtype)
#print(type(Known1),Known1.shape,Known1,Known1.dtype)
# print((np.round(data,decimals=2)==np.round(data1,decimals=2)).all())
# print('1111', np.where(temp != 0))
#assert 0

data  = torch.tensor(data,dtype=torch.float32).cuda()
Known  = torch.tensor(Known,dtype=torch.int64).cuda()

#print(type(data),data.shape,data,data.dtype)
#print(type(Known),Known.shape,Known,Known.dtype)
#assert 0

opts={'maxit':10000,'Zfull':1,'DoQR':1,'print':0,'est_rank':2}



U,V ,_= lmafit_mc_adp(size_temp1[0],size_temp1[1]*size_temp1[2],1,Known,data,opts)
print(U,U.shape)
print(V,V.shape)
assert 0
V = np.transpose(V)

meas_id=find_index(rmask)
meas=find_num(rval,meas_id )

X=admm_hankel(U,V,meas,meas_id,mu,muiter,Nimg,H,Hi);

