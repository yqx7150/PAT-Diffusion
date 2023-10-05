Input_path = '/home/liuqg/桌面/test1.30/input/';  
Output_path='/home/liuqg/桌面/test1.30/output';
namelist = dir(strcat(Input_path,'*.png'));  %获得文件夹下所有的 .jpg图片
len = length(namelist);
%%kwave
parallel.gpu.enableCUDAForwardCompatibility(true);

if(~exist('kspaceFirstOrder3D.m', 'file'))
       error('kWave toolbox must be on the path to execute this part of the code') 
end
load('/home/liuqg/wgj/diffu1/st_1000.mat')
dataCast    =  'gpuArray-single';
    smoothP0    = true;
    codeVersion = 'Matlab'; 

    inputArgs   = {'PMLSize', 20, 'DataCast', dataCast, 'Smooth', smoothP0,...
        'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};
    
    % define function handles for forward and adjoint operator
    A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
    Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});



for i = 1:len
    name=namelist(i).name;  %namelist(i).name; %这里获得的只是该路径下的文件名
    I=imread(strcat(Input_path, name)); %图片完整的路径名
    pic = double(padarray(I,[92 92]));    
    sensordata1=A(pic);
    x = Aadj(sensordata1);
    xi=x(93:1:348,93:1:348);
    %xi=(xi-min(min(xi)))/(max(max(xi)-min(min(xi))))
    imwrite(xi,[Output_path,'xueguan',int2str(i),'.png']); %完整的图片存储的路径名  并将整形的数字 
                                                        
end