clear all;

input_image = imread("test.jpg");
input_image = im2double(input_image);

lambda = 0.1;         
sigma = 0.3;     
p = 0.5;
tau = 1;


tic;
smoothed_image = image_adapt_smooth(input_image,lambda, p, tau, sigma);
toc;



figure;
imshow(input_image);





