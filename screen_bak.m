function varargout = screen_bak(varargin)
% SCREEN_BAK MATLAB code for screen_bak.fig
%      SCREEN_BAK, by itself, creates a new SCREEN_BAK or raises the existing
%      singleton*.
%
%      H = SCREEN_BAK returns the handle to a new SCREEN_BAK or the handle to
%      the existing singleton*.
%
%      SCREEN_BAK('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SCREEN_BAK.M with the given input arguments.
%
%      SCREEN_BAK('Property','Value',...) creates a new SCREEN_BAK or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before screen_bak_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to screen_bak_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help screen_bak

% Last Modified by GUIDE v2.5 08-Dec-2024 10:11:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @screen_bak_OpeningFcn, ...
                   'gui_OutputFcn',  @screen_bak_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before screen_bak is made visible.
function screen_bak_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to screen_bak (see VARARGIN)

% Choose default command line output for screen_bak
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes screen_bak wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = screen_bak_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%%
%%导入图片
% --- Executes on button press in load_in.
function load_in_Callback(hObject, eventdata, handles)
global img; 
img=imread('bird.jpg');
axes(handles.original);imshow(img);
axes(handles.final);imshow(img);





%%
%%规定化
% --- Executes on button press in histogram.
function histogram_Callback(hObject, eventdata, handles)
global grayimg;
%直方图均衡化
histogram = imhist(grayimg);
    % 计算累积直方图
    cumHist = cumsum(histogram) / numel(grayimg);
    % 计算均衡化后的像素值
    img_eq = uint8(255 * cumHist(double(grayimg) + 1));

%直方图匹配
target = imread('target.jpg');
target_gray = rgb2gray(target);
    % 计算直方图
histImg = imhist(grayimg);
histTarget = imhist(target_gray);
    % 计算累积直方图
cumHistImg = cumsum(histImg) / numel(grayimg);
cumHistTarget = cumsum(histTarget) / numel(target_gray);
    % 初始化匹配后的图像
img_matched = zeros(size(grayimg));
    % 直方图匹配
for level = 0:255
    % 找到最接近的匹配值
    [minVal, idx] = min(abs(cumHistImg(level+1) - cumHistTarget));
    img_matched(grayimg == level) = idx - 1; % 确保索引从0开始
end
    % 转换为uint8并显示
img_matched = uint8(img_matched);

axes(handles.original);imshow(img_eq);
axes(handles.final);imshow(img_matched);



%%
%%灰度化
% --- Executes on button press in gray.
function gray_Callback(hObject, eventdata, handles)
global img;
global grayimg;
[width,height,channels]=size(img);  %返回图像的宽度、高度和颜色通道数
if(channels==3)
    r_matrix=img(:,:,1);
    g_matrix=img(:,:,2);
    b_matrix=img(:,:,3);
    Grayp2_matrix=zeros(width,height);%初始化一个与图像相同尺寸的矩阵Grayp2_matrix，用于存储灰度图像的像素值，初始值都设为0。
    for i=1:width
        for j=1:height%按行列便利，根据通道逐个加权
            Grayp2_matrix(i,j)=r_matrix(i,j).*0.299+g_matrix(i,j).*0.587+b_matrix(i,j).*0.114;
        end
    end
    Grayp2 = uint8(Grayp2_matrix);%数据类型转换为uint8（无符号8位整数），这是图像处理中常用的数据类型，范围从0到255。
else
    Grayp2 = img;%如果已经是灰度图像就直接使用
end
    grayimg = Grayp2;

axes(handles.final);imshow(grayimg);




%%
%%线性、非线性对比度增强
% --- Executes on button press in liner_variation.
function liner_variation_Callback(hObject, eventdata, handles)
set(handles.liner_variation,'value',1);
set(handles.nonliner_variation,'value',0);

% --- Executes on button press in nonliner_variation.
function nonliner_variation_Callback(hObject, eventdata, handles)
set(handles.liner_variation,'value',0);
set(handles.nonliner_variation,'value',1);

% --- Executes on button press in contrast_enhancement.
function contrast_enhancement_Callback(hObject, eventdata, handles)
global grayimg;
if get(handles.liner_variation,'value')
		
		    img_double = double(grayimg);
            slope = 1.9;  % 斜率
            intercept = 2;  % 截距
            img_enhanced = slope * img_double + intercept;
    
            % 限制像素值在0和255之间（uint8图像的范围）
            img_enhanced = min(max(img_enhanced, 0), 255);
    
            % 将结果转换回uint8类型
            img_enhanced = uint8(img_enhanced);
            axes(handles.final);imshow(img_enhanced);
		
elseif get(handles.nonliner_variation,'value')
		    
        %对数变化
		    img_double = double(grayimg);
            img_enhanced1 = log(1 + img_double) / log(1 + max(img_double(:)));
            % 将结果转换回uint8类型
            img_enhanced1 = uint8(255 * img_enhanced1);
            
        %指数变化
            gamma = 0.8;  % 指数
            img_enhanced = (img_double / double(max(img_double(:))) ^ gamma);
            % 将结果转换回uint8类型
            img_enhanced = uint8(255 * img_enhanced);
		axes(handles.original);imshow(img_enhanced1);
        axes(handles.final);imshow(img_enhanced);
end




%%
%%缩放、旋转
% --- Executes on button press in zoom.
function zoom_Callback(hObject, eventdata, handles)
global img;
scaleFactor = handles.scaleFactor;
img_double=double(img);
 scaledImg = customImresize(img_double, scaleFactor);
 scaledImg = uint8(scaledImg);
axes(handles.final);imshow(scaledImg);
function scaledImg = customImresize(img, scaleFactor)
   % 获取原始图像的尺寸
    [rows, cols, ~] = size(img);
    
    % 计算缩放后的尺寸
    newRows = round(rows * scaleFactor);
    newCols = round(cols * scaleFactor);
    
    % 初始化缩放后的图像
    scaledImg = zeros(newRows, newCols, 3);
    
    % 计算缩放因子的倒数
    invScaleFactor = 1 / scaleFactor;
    
    % 对每个像素进行双三次插值
    for i = 1:newRows
        for j = 1:newCols
            % 计算原始图像中的对应位置
            origRow = (i - 0.5) * invScaleFactor + 0.5;
            origCol = (j - 0.5) * invScaleFactor + 0.5;
            
            % 确保坐标在原始图像范围内
            if origRow >= 1 && origRow <= rows && origCol >= 1 && origCol <= cols
                % 双三次插值
                x1 = floor(origRow);
                x2 = ceil(origRow);
                y1 = floor(origCol);
                y2 = ceil(origCol);
                
                Q11 = img(x1, y1, :);
                Q12 = img(x1, y2, :);
                Q21 = img(x2, y1, :);
                Q22 = img(x2, y2, :);
                
                w1 = origRow - x1;
                w2 = x2 - origRow;
                w3 = origCol - y1;
                w4 = y2 - origCol;
                
                interpolatedValue = (w2 * (w3 * Q11 + w4 * Q12) + w1 * (w3 * Q21 + w4 * Q22)) / (w1 + w2) / (w3 + w4);
                
                scaledImg(i, j, :) = interpolatedValue;
            else
                % 如果超出范围，可以设置为边界值或其他处理方式
                scaledImg(i, j, :) = img(max(1, min(rows, round(origRow))), max(1, min(cols, round(origCol))), :);
            end
        end
    end
function zoom_percentage_Callback(hObject, eventdata, handles)


userInput = get(handles.zoom_percentage, 'String');
  try
        scaleFactor = str2double(userInput);
        if isnan(scaleFactor) || scaleFactor <= 0
            error('请输入一个有效的正数放大倍数！');
        end
        
        % 将缩放因子存储在handles结构体中
        guidata(hObject, handles); % 确保handles是最新的
        handles.scaleFactor = scaleFactor;
        guidata(hObject, handles); % 更新handles结构体
        
        % 调用缩放回调函数
        zoom_Callback(hObject, eventdata, handles);
        
    catch
        error('请输入一个有效的正数放大倍数！');
  end
% --- Executes during object creation, after setting all properties.
function zoom_percentage_CreateFcn(hObject, eventdata, handles)


%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in rotate.
function rotate_Callback(hObject, eventdata, handles)
global img;
img_double = double(img);
angle = handles.angle;
axPos = get(handles.final, 'Position');
axWidth = axPos(3);
axHeight = axPos(4);
axCenterX = axPos(1) + axWidth / 2;
axCenterY = axPos(2) + axHeight / 2;

% 将图像中心移动到axes中心
img_double = imtranslate(img_double, [-axCenterX axCenterY], 'FillValues', 0);

rotatedImg = customRotate(img_double, angle);

% 将旋转后的图像中心移回原始图像中心
rotatedImg = imtranslate(rotatedImg, [axCenterX -axCenterY], 'FillValues', 0);
rotatedImg = uint8(rotatedImg);
axes(handles.final);
imshow(rotatedImg);
% 调整axes的位置使其居中

    function rotatedImg = customRotate(img, angle)
       % 将角度转换为弧度
    theta = deg2rad(angle);

    % 获取图像尺寸
    [height, width, numChannels] = size(img);

    % 计算旋转矩阵
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    % 计算旋转后的图像中心点
    center = [width/2; height/2];
    % 创建一个新的空白图像
    newImg = zeros(height, width, numChannels, 'like', img);
    % 应用插值方法进行像素映射
    for i = 1:height
        for j = 1:width
            % 反向映射到原始图像坐标
            originalCoord = R' * ([j - center(1); i - center(2)]);
            x = originalCoord(1) + center(1);
            y = originalCoord(2) + center(2);
            % 确保坐标在图像边界内
            x1 = max(min(floor(x), width), 1);
            x2 = max(min(ceil(x), width), 1);
            y1 = max(min(floor(y), height), 1);
            y2 = max(min(ceil(y), height), 1);

            % 双线性插值
            dx = x - x1;
            dy = y - y1;
            dx1 = 1 - dx;
            dy1 = 1 - dy;
            q11 = img(y1, x1, :);
            q12 = img(y1, x2, :);
            q21 = img(y2, x1, :);
            q22 = img(y2, x2, :);

            value = dx1 * dy1 * q11 + dx * dy1 * q12 + dx1 * dy * q21 + dx * dy * q22;
            newImg(i, j, :) = value;
        end
    end
    rotatedImg = newImg;%裁剪为原来尺寸，确保能在原来的图像框显示


    
function edit2_Callback(hObject, eventdata, handles)
% 获取用户输入的角度
    userInput = get(handles.edit2, 'String');
    
    % 尝试将输入转换为双精度浮点数
    angle = str2double(userInput);
    
    % 检查角度是否有效
    if isnan(angle) || angle < 0
        error('请输入一个有效的正数角度！');
    end
    
    % 将角度存储在handles结构体中
    handles.angle = angle;
    guidata(hObject, handles); % 更新handles结构体
    
    % 调用旋转函数
    rotate_Callback(hObject, eventdata, handles);
      
        
        % --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%
%%空域，频域，加噪滤波
% --- Executes on button press in spatial_domain.
function spatial_domain_Callback(~, eventdata, handles)
set(handles.spatial_domain,'value',1);
set(handles.frequence_domain,'value',0);

% --- Executes on button press in frequence_domain.
function frequence_domain_Callback(hObject, eventdata, handles)
set(handles.spatial_domain,'value',0);
set(handles.frequence_domain,'value',1);

% --- Executes on button press in fliter.
%%空域，频域，加噪滤波
function fliter_Callback(hObject, eventdata, handles)
global grayimg;
%添加高斯噪声
 img_double = double(grayimg);

            % 定义噪声方差和生成噪声
            noise_variance = 36; 
            % 使用 img_double 的尺寸生成噪声
            noise = noise_variance * randn(size(img_double), 'like', img_double);

            % 将噪声添加到图像上
            noisy_img = img_double + noise;

            % 限制像素值在0和255之间并转换回uint8类型
            noisy_img = uint8(max(min(noisy_img, 255), 0));

%空域均值滤波
if get(handles.spatial_domain,'value')
   % 初始化滤波器核
   filter_size = 3;
   filter_kernel = ones(filter_size) / (filter_size^2);
   % 应用滤波器核
   % 对 noisy_img 进行卷积
   img_filtered = conv2(noisy_img, filter_kernel, 'same');

   axes(handles.final);imshow(uint8(img_filtered));%控制格式，像素再输出

%频域低通滤波		
elseif get(handles.frequence_domain,'value')
		
        cutoff_frequency = 30;  % 尝试一个更高的截止频率，10产生的图片纯黑色

        % 对图像进行傅里叶变换
        F = fft2(img_double);
        [rows, cols] = size(F);
        F_shifted = fftshift(F);  % 将零频分量移到频谱中心

        % 创建低通滤波器
        [u, v] = meshgrid(-cols/2:cols/2-1, -rows/2:rows/2-1);
        filter_kernel = sqrt(u.^2 + v.^2) <= cutoff_frequency;

        % 应用波器
        F_filtered = F_shifted .* filter_kernel;

        % 逆傅里叶变换
        img_filtered = ifft2(ifftshift(F_filtered));  % 将频谱移回原点

        % 转换回uint8类型并显示结果
        img_filtered = uint8(255 * (abs(img_filtered) - min(abs(img_filtered(:)))) / (max(abs(img_filtered(:))) - min(abs(img_filtered(:)))));

       
        axes(handles.final);imshow(img_filtered);
end




%%
%%robert，prewitt，sobel、边缘提取
% --- Executes on button press in robert.
function robert_Callback(hObject, eventdata, handles)
set(handles.robert,'value',1);
set(handles.prewitt,'value',0);
set(handles.sobel,'value',0);
set(handles.lapras,'value',0);

% --- Executes on button press in prewitt.
function prewitt_Callback(hObject, eventdata, handles)
set(handles.robert,'value',0);
set(handles.prewitt,'value',1);
set(handles.sobel,'value',0);
set(handles.lapras,'value',0);

% --- Executes on button press in sobel.
function sobel_Callback(hObject, eventdata, handles)
set(handles.robert,'value',0);
set(handles.prewitt,'value',0);
set(handles.sobel,'value',1);
set(handles.lapras,'value',0);

function lapras_Callback(hObject, eventdata, handles)
set(handles.robert,'value',0);
set(handles.prewitt,'value',0);
set(handles.sobel,'value',0);
set(handles.lapras,'value',1);

% --- Executes on button press in edge_extraction.
function edge_extraction_Callback(hObject, eventdata, handles)
global grayimg;
img_double = double(grayimg);
[rows, cols] = size(img_double);

%%robert边缘提取
if get(handles.robert,'value')
    robert_kernel = [1 0; 0 -1];
    edges_robert = zeros(rows, cols);
      for i = 2:rows-1
        for j = 2:cols-1
           gradient_robert = abs(robert_kernel(1,1)*img_double(i-1,j-1) + robert_kernel(1,2)*img_double(i-1,j) + ...
                robert_kernel(2,1)*img_double(i,j-1) + robert_kernel(2,2)*img_double(i,j));
            edges_robert(i, j) = gradient_robert;
        end
      end


     maxGradientMax = max(edges_robert(:));
    if maxGradientMax == 0
        maxGradientMax = eps; % 如果最大梯度值为0，则设置为eps
    end

     % 归一化边缘强度矩阵，使用点除确保操作应用于每个元素
    edges_robert_normalized = (edges_robert ./ maxGradientMax) * 255;
    % 检查是否有 NaN 值，并替换它们
    edges_robert_normalized(isnan(edges_robert_normalized)) = 0;
    % 确保归一化后的值在0到255的范围内
    edges_robert_normalized(edges_robert_normalized < 0) = 0;
    edges_robert_normalized(edges_robert_normalized > 255) = 255;
    % 转换为 uint8
    edges_robert_normalized = uint8(edges_robert_normalized);

    axes(handles.final);imshow(edges_robert_normalized);

%%prewitt边缘提取
elseif get(handles.prewitt,'value')
    prewitt_kernel = [-1 0 1; -1 0 1];
    edges_prewitt = zeros(rows, cols);
    for i = 2:rows-1
    for j = 2:cols-1
        gradient_prewitt = abs(prewitt_kernel(1,1)*img_double(i-1,j-1) + prewitt_kernel(1,2)*img_double(i-1,j) + ...
                                     prewitt_kernel(1,3)*img_double(i-1,j+1) + prewitt_kernel(2,1)*img_double(i,j-1) + ...
                                     prewitt_kernel(2,2)*img_double(i,j) + prewitt_kernel(2,3)*img_double(i,j+1));
        edges_prewitt(i, j) = gradient_prewitt;
    end
end

    edges_prewitt = uint8(edges_prewitt * 255 ./ max(edges_prewitt(:)));
    axes(handles.final);imshow(edges_prewitt);

%%sobel边缘提取
elseif get(handles.sobel,'value')
    sobel_kernel = [-1 0 1; -2 0 2; -1 0 1];
    edges_sobel = zeros(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            gradient_sobel = abs(sobel_kernel(1,1)*img_double(i-1,j-1) + sobel_kernel(1,2)*img_double(i-1,j) + ...
                sobel_kernel(1,3)*img_double(i-1,j+1) + sobel_kernel(2,1)*img_double(i,j-1) + ...
                sobel_kernel(2,2)*img_double(i,j) + sobel_kernel(2,3)*img_double(i,j+1) + ...
                sobel_kernel(3,1)*img_double(i+1,j-1) + sobel_kernel(3,2)*img_double(i+1,j) + ...
                sobel_kernel(3,3)*img_double(i+1,j+1));
            edges_sobel(i, j) = gradient_sobel;
        end
    end

    edges_sobel = uint8(edges_sobel * 255 ./ max(edges_sobel(:)));
    axes(handles.final);imshow(edges_sobel);

%% laplacian、边缘提取
elseif get(handles.lapras,'value')
    laplacian_kernel = [0 1 0; 1 -4 1; 0 1 0];
    edges_laplacian = zeros(rows, cols);
    for i = 2:rows-1  % 避免边界像素
        for j = 2:cols-1
            % 提取图像的邻域
            neighborhood = img_double(i-1:i+1, j-1:j+1);

            % 计算卷积（梯度）
            gradient_laplacian = sum(sum(laplacian_kernel .* neighborhood));

            % 存储计算结果
            edges_laplacian(i, j) = gradient_laplacian;
        end
    end
    edges_laplacian = uint8(edges_laplacian * 255 ./ max(edges_laplacian(:)));
    axes(handles.final);imshow( edges_laplacian);
end


%%
%%HOG,LBP、目标提取、特征提取
% --- Executes on button press in LBP.
function LBP_Callback(hObject, eventdata, handles)
set(handles.LBP,'value',1);
set(handles.HOG,'value',0);

% --- Executes on button press in HOG.
function HOG_Callback(hObject, eventdata, handles)
set(handles.LBP,'value',0);
set(handles.HOG,'value',1);


%%HOG,LBP、目标提取、特征提取
% --- Executes on button press in feature_extraction.
function feature_extraction_Callback(hObject, eventdata, handles)
global grayimg;
if get(handles.LBP,'value')
[rows, cols] = size(grayimg);
radius = 1; % 3x3邻域的半径
lbpFeatures = zeros(rows-2*radius, cols-2*radius);

for i = radius+1:rows-radius
    for j = radius+1:cols-radius
        binaryPattern = 0;
        for m = -radius:radius
            for n = -radius:radius
                if m == 0 && n == 0
                    continue;
                end
                if grayimg(i+m, j+n) >= grayimg(i, j)
                    binaryPattern = binaryPattern + 2^((m+1)*2 + (n+1));
                end
            end
        end
        % 计算模式的均匀性
        numTransitions = sum(binaryPattern == 1 & floor(binaryPattern/2) == 0) + ...
                         sum(binaryPattern == 2 & floor(binaryPattern/4) == 0);
        if numTransitions <= 2
            lbpFeatures(i-radius, j-radius) = binaryPattern;
        else
            lbpFeatures(i-radius, j-radius) = 0; % 或者设置为一个特定的值表示非均匀模式
        end
    end
end

% 接下来的步骤与之前相同：归一化、直方图均衡化、颜色映射和显示
% 归一化LBP特征图
lbpFeaturesNorm = mat2gray(lbpFeatures);
% 直方图均衡化
lbpFeaturesEq = histeq(lbpFeaturesNorm);
% 将LBP特征图转换为颜色图像，使用不同的颜色映射
lbpColorImage = ind2rgb(uint8(lbpFeaturesEq * 255), colormap('hsv'));
 axes(handles.final);imshow( lbpColorImage);

elseif get(handles.HOG,'value')

 img_gray = imadjust(grayimg, [], [], 0.5); % 增强对比度并转换为灰度图像
    img_gray = double(img_gray);


 % 设置最大图像尺寸以防止内存溢出
    maxDim = 128; % 或者其他适合你系统的值
    [m, n] = size(img_gray);
    if m > maxDim || n > maxDim
        img_gray = imresize(img_gray, [min(m, maxDim) min(n, maxDim)], 'nearest');
    end
    [m, n] = size(img_gray);

    % 伽马校正
    img_gray = sqrt(img_gray);

    % 使用imgradient计算梯度和方向
    [Gmag, Gdir] = imgradient(img_gray); % 梯度幅值和方向

    % 将角度转换到0-360范围
    Gdir = mod(Gdir * 180 / pi + 180, 360);

    orient = 9; % 方向直方图的方向个数
    jiao = 360 / orient; % 每个方向包含的角度数

    % 计算单元格数量并确保它们是整数
    step = 8; % 8x8个像素作为一个cell
    numCellsX = floor(n / step);
    numCellsY = floor(m / step);
    % 初始化Cell直方图
    Hist = zeros(numCellsY, numCellsX, orient);
    % 计算每个像素点的bin位置
    bins = ceil((Gdir + 180) / jiao);
    bins(bins < 1) = 1;
    bins(bins > orient) = orient;

    % 直接在二维数组上累积梯度幅值
    for k = 1:orient
        mask = bins == k;
        maskedGmag = Gmag .* double(mask);
        
        % 累积梯度幅值到对应的cell和方向
        for row = 1:m
            for col = 1:n
                cellRow = floor((row - 1) / step) + 1;
                cellCol = floor((col - 1) / step) + 1;
                
                if cellRow >= 1 && cellRow <= numCellsY && ...
                   cellCol >= 1 && cellCol <= numCellsX && ...
                   maskedGmag(row, col) > 0
                    Hist(cellRow, cellCol, k) = Hist(cellRow, cellCol, k) + maskedGmag(row, col);
                end
            end
        end
    end

    % 划分block，求block的特征值（向量化操作）
    blockSize = 2;
    numBlocksX = numCellsX - blockSize + 1;
    numBlocksY = numCellsY - blockSize + 1;

    % 预分配空间
    featureVec = zeros(numBlocksY * numBlocksX, orient * blockSize^2);

    for i = 1:numBlocksY
        for j = 1:numBlocksX
            blockFeatures = Hist(i:i+blockSize-1, j:j+blockSize-1, :);
            blockFeatures = blockFeatures(:)';
            if sum(blockFeatures) > 0
                blockFeatures = blockFeatures / norm(blockFeatures); % L2归一化
            end
            featureVec((i-1)*numBlocksX + j, :) = blockFeatures;
        end
    end

    % 返回HOG特征向量
    hogFeature = featureVec;


    axes(handles.final);imshow( hogFeature);
end

% --- Executes on button press in target_extraction.
function target_extraction_Callback(hObject, eventdata, handles)

global grayimg;
img_gray=grayimg;
 % 二值化图像
    bw = imbinarize(img_gray); % 假设目标是亮区域，背景是暗区域

    % 定义3x3的结构元素
    se = ones(3); % 这里简化为3x3结构元素

    % 腐蚀操作
    erodedImage = erode(bw, se);

    % 膨胀操作
    dilatedImage = dilate(bw, se);
     axes(handles.original);imshow( erodedImage);
      axes(handles.final);imshow( dilatedImage);
% 腐蚀函数
function erodedImage = erode(img, se)
    [rows, cols] = size(img);
    erodedImage = zeros(size(img));
    padSize = (max(se(:)) - 1) / 2; % 计算填充大小
    pad_img = padarray(img, padSize, 0, 'both'); % 填充0以处理边界
    
    for i = 1:rows
        for j = 1:cols
            neighborhood = pad_img(i:i+se(1)-1, j:j+se(2)-1);
            erodedImage(i, j) = min(neighborhood(:));
        end
    end


% 膨胀函数
function dilatedImage = dilate(img, se)
    [rows, cols] = size(img);
    dilatedImage = zeros(size(img));
    padSize = (max(se(:)) - 1) / 2; % 计算填充大小
    pad_img = padarray(img, padSize, 0, 'both'); % 填充0以处理边界
    
    for i = 1:rows
        for j = 1:cols
            neighborhood = pad_img(i:i+se(1)-1, j:j+se(2)-1);
            dilatedImage(i, j) = max(neighborhood(:));
        end
    end

% 填充函数
function paddedImg = padarray(img, padSize, padValue, direction)
    [padRows, padCols] = deal(padSize, padSize); % 确保填充大小是标量
    newRows = size(img, 1) + 2 * padRows;
    newCols = size(img, 2) + 2 * padCols;
    paddedImg = padValue * ones(newRows, newCols);
    paddedImg(padRows+1:padRows+size(img, 1), padCols+1:padCols+size(img, 2)) = img;
