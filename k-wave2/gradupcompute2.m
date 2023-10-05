
function gradup= gradupcompute2(xi,y0)

xi = double(padarray(xi,[92,92]));

    if(~exist('kspaceFirstOrder3D.m', 'file'))
       error('kWave toolbox must be on the path to execute this part of the code') 
    end
    
    % load struct that contains setting information and subSampling mask
    load('/home/liuqg/wgj/diffu2/limit70_2.25.mat')
    
%     recSize = [kgrid.Nx, kgrid.Ny, kgrid.Nz];

    % for these settings, see kWave documentation. 


    % for an explanation of the options, see kWaveWrapper.m
    dataCast    =  'gpuArray-single';
    smoothP0    = true;
    codeVersion = 'Matlab'; 

    inputArgs   = {'PMLSize', 20, 'DataCast', dataCast, 'Smooth', smoothP0,...
        'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};
    
    % define function handles for forward and adjoint operator
    A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
    Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});
    forward=A(xi);
    %disp(size(sensor_data1));
    gradUp0= Aadj(forward-y0);
    
    %gradup=gradUp0(51:1:562,51:1:562);
    gradup=gradUp0(93:1:348,93:1:348);

    clearvars -EXCEPT gradup    



end
