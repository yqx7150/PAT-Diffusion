import os

import matplotlib.pyplot as plt
from matplotlib.image import imread
 


def get_allfile(path):  # 获取所有文件
    all_file = []
    s_file = []
    for f in os.listdir(path):  #listdir返回文件中所有目录
        f_name = os.path.join(path, f)
        all_file.append(f_name)
        print(all_file)
        iname = ''.join(all_file)
        img = imread(iname)
        plt.imshow(img)
 
        plt.show()
        #assert 0
        
        
        
    return all_file
    
all_file=get_allfile('./lzdata/CImageNet400_Test')  #tickets要获取文件夹名
#print(all_file)


