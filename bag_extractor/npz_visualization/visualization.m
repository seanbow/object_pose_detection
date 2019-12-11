npz_folder = '/home/kodlab/saved_images/';
addpath('npy-matlab/npy-matlab')

% unzip all files
npz_files = dir([npz_folder '*.npz']);
for i = 1:numel(npz_files)
    unzip([npz_folder npz_files(i).name], [npz_folder npz_files(i).name(1:end-4)])
end

% intrinsics
K = [679.6778564453125, 0, 637.9580078125;
0, 679.6778564453125, 354.5067138671875;
0, 0, 1];

% 3D keypoints
load('cad.mat')
kpt = [kpt_1; kpt_2; kpt_3; kpt_4; kpt_5; kpt_6; kpt_7; kpt_8; kpt_9; kpt_10];

% visualization
counter = 0;
npz_files = dir([npz_folder '*.npz']);
for i = 1:10:numel(npz_files)
    i
    I = imread([npz_folder npz_files(i).name(1:end-4) '.jpg']);
    ori0 = readNPY([npz_folder npz_files(i).name(1:end-4) '/quat.npy'])';
    pos = readNPY([npz_folder npz_files(i).name(1:end-4) '/trans.npy']);
    ori = ori0;
    ori(1) = ori0(4);
    ori(2:4) = ori0(1:3); % w,x,y,z
    ori_rotm = quat2rotm(ori);
    d3 = K*(ori_rotm*[kpt']+pos);
    pix = d3(1:2,:)./d3(3,:);
    imshow(I)
    hold on;
    scatter(pix(1,1:10),pix(2,1:10),200,'b','*')
    %scatter(pix(1,10),pix(2,10),'b','*')
    pause(1)
    F = getframe;
    imwrite(imresize(F.cdata, [660 1172]), ['~/output/' num2str(counter,'%03d') '.jpg']);
    counter = counter + 1;
end