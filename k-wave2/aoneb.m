
parallel.gpu.enableCUDAForwardCompatibility(true);
Input_path = '/home/liuqg/wgj/diffu2/onback/input/';  
Output_path='/home/liuqg/wgj/diffu2/onback/output/';
namelist = dir(strcat(Input_path,'*.png'));  %获得文件夹下所有的 .jpg图片
len = length(namelist);
for i = 1:len
    name=namelist(i).name;  %namelist(i).name; %这里获得的只是该路径下的文件名
    pic=imread(strcat(Input_path, name)); %图片完整的路径名
    sensor1 = forward2(pic);
    pic2 = backward2(sensor1);
    immax=max(max(pic2));
    immin = min(min(pic2));
    pic3=pic2;
    pic3=(pic3-immin)/(immax-immin);
    pic3 = pic3*255;
    [peaksnr,snr] = psnr(double(pic3),double(pic));
    fid = fopen('qqqq.txt','a+');
    fprintf(fid,'%.4f\n',peaksnr);
    pic2=(pic2-immin)/(immax-immin);
    imwrite(pic2,[Output_path,'rec',int2str(i),'_l60.png']); %完整的图片存储的路径名  并将整形的数字 
    
                                                        
end