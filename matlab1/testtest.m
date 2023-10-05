
%a=imread('02.png');
for i=1:15

a=forward1(a);


a=backward1(a);
end

figure;imagesc();colormap(gray);


e=gradupcompute(c,b);