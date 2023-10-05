function gradUp= Gradup(xi)

%  Script for calling the kWave operattor for learned 3D photoacoustic
%  imaging. This script initialises the model in k-wave and
%  evaluates the gradients.
% 
%  This is accompanying code for: Hauptmann et al., Model based learning for 
%  accelerated, limited-view 3D photoacoustic tomography, 
%  https://arxiv.org/abs/1708.09832
% 
%  written by Felix Lucka and Andreas Hauptmann, January 2018
%  ==============================================================================

%This part is called as initialisation 
% if(compInit)

    % check if kWave is on the path
    if(~exist('kspaceFirstOrder3D.m', 'file'))
       error('kWave toolbox must be on the path to execute this part of the code') 
    end
    
    % load struct that contains setting information and subSampling mask
    load('C:\Users\zzs\Desktop\experiment\Photoacoustic\AandAdj\前向\k-Wave\setting10.mat')
    
%     recSize = [kgrid.Nx, kgrid.Ny, kgrid.Nz];

    % for these settings, see kWave documentation. 
%     medium               = [];
%     medium.sound_speed   = 1580; % [m/s] speed of sound
%     sensor               = [];      
%     sensor.mask          = false([kgrid.Nx, kgrid.Ny]);
%   sensor.mask(1, :, :) = subSamplingMask;

    % for an explanation of the options, see kWaveWrapper.m
    dataCast    =  'gpuArray-single';
    smoothP0    = true;
    codeVersion = 'Matlab'; 

    inputArgs   = {'PMLSize', 20, 'DataCast', dataCast, 'Smooth', smoothP0,...
        'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};
    
    % define function handles for forward and adjoint operator
    A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
    Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});
    sensor_data1=A(xi);
%     gradUp= Aadj(A(xi)-y0);
   gradUp=sensor_data1;
    
end
