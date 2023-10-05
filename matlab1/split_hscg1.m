function x = split_hscg1(x,y,u,alpha,niter)

clear grad d gradold Hd step
% Conjugate HS gradient method for solving min _f ||Hf-g||_2^2+alpha_1||f-u^(i-1)||_2^2
% x the restoreed sinogram, y measure data (raw data)
% x = f, y=g
%%%%%%%% 
%%%%%%

grad=zeros(256,256);


[rows, cols] = size(x);
% x = x(:);
% y = y(:);
% u = u(:);



for iter = 1:niter%niter
    grad=double(backward2(forward2(x)-y))+alpha * (x - u);
    grad=grad(:);
%     grad=double(Aadj(A(x)-y))+alpha * (x - u);
%     grad = double(H' * (H * x - y) + alpha * (x - u)); % gradient
    if iter ==1
        d = -grad;
    else
        d = -grad + (grad'*(grad-gradold))/(d'*(grad-gradold))*d;
    end
    gradold = grad;
    d=reshape(d,rows, cols);
    Hd = double(forward2(d));
    d=d(:);
    Hd=Hd(:);
%     Hd = double(H*d);
    step = -(grad'*d)/(Hd'*Hd + d'*d*alpha); % optimal stepsize
    x = x(:);
    x = x + step * d;
    x = reshape(x,rows, cols);
    
    
end
    clearvars -EXCEPT x

