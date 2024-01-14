clear all;close all;

varn = 100;
clean_folder = '000';
noised_folder = 'noised000var100'; % 图片所在文件夹
output_folder = '000_vbm4d_var100_est';
num_images = 100; % 图片数量
% 假设图片大小为 MxN，这里需要根据实际情况调整
M = 720; % 图片的高度
N = 1280; % 图片的宽度
% 初始化数组来存储图片，第四维是图片的数量
images = zeros(M, N, 3, num_images, 'single');

filename = fullfile(clean_folder, sprintf('%08d.png', 0));
% 读取图片
img = imread(filename);
% 存储图片
images(:, :, :, 1) = img;

% 循环读取每张图片
for i = 1:(num_images-1)
    % 构造文件名
    filename = fullfile(noised_folder, sprintf('%08d.png', i));
    % 读取图片
    img = imread(filename);
    % 存储图片
    images(:, :, :, i+1) = img;
end


% Modifiable parameters
sigma = -1;      % Noise standard deviation. it should be in the same
% intensity range of the video
profile = 'np';      % V-BM4D parameter profile
%  'lc' --> low complexity
%  'np' --> normal profile
do_wiener = 1;       % Wiener filtering
%   1 --> enable Wiener filtering
%   0 --> disable Wiener filtering
sharpen = 1;         % Sharpening
%   1 --> disable sharpening
%  >1 --> enable sharpening
deflicker = 1;       % Deflickering
%   1 --> disable deflickering
%  <1 --> enable deflickering
verbose = 1;         % Verbose mode


% V-BM4D filtering
tic;
S = 255;
images = cast(images, 'single')/S;
sigma = sigma/S;
denoised_images = vbm4d(images, sigma, profile, do_wiener, sharpen, deflicker, verbose );
time = toc;

% 循环遍历并保存每张图片
for i = 1:num_images
    % 提取单张图片
    img = denoised_images(:, :, :, i);

    % 构建文件名
    filename = fullfile(output_folder, sprintf('%08d.png', i-1));

    % 保存图片
    imwrite(img, filename);
end







