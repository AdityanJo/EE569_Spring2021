%
%Adityan Jothi
%USC ID 8162222801
%jothi@usc.edu
%
function problem2()
    src_image = imread('composite_updated.png');
    [h w] = size(src_image);
    
    window_size = 11;
    filters = generateFilters();
    feature_space = zeros(h, w, 25);
    
    src_image = (src_image - mean(src_image,'all'));
    for j=1:25
        response = convolve5x5Wrapper(src_image, filters(:,:,j));
        avgEn = calcAverageEnergyWindowed(response, window_size);
        feature_vectors(:,:,j) = avgEn;
    end
    %for j=1:25
    %    feature_vectors(:,:,j) = feature_vectors(:,:,j)./feature_vectors(:,:,1);
    %end
    %feature_vectors(:,:,1) =[];
    flattened_feature_vectors = zeros(h*w, 25);
    idx = 1;
    for i=1:h
        for j =1:w
            flattened_feature_vectors(idx, :) = feature_vectors(i,j,:);
            idx = idx+1;
        end
    end
    [idx, centroids] = kmeans(flattened_feature_vectors, 5);
    segmentation_map = zeros(h,w);
    jdx=1;
    colors = [0, 63, 127, 191, 255];
    for i=1:h
        for j=1:w
            segmentation_map(i,j) = colors(idx(jdx));
            jdx = jdx+1;
        end
    end
    red_fv_coeff = pca(flattened_feature_vectors,'NumComponents', 3);
    red_feat_vec = matmul(flattened_feature_vectors, red_fv_coeff);
    [idx, centroids] = kmeans(red_feat_vec, 5);
    figure;
    image(segmentation_map);
    title('25-D segmentation');
    figure;
    image(enhanceMap(segmentation_map,colors));
    title('25-D segmentation after postprocessing');
    
    red_segmentation_map = zeros(h,w);
    jdx=1;
    colors = [0, 63, 127, 191, 255];
    for i=1:h
        for j=1:w
            %closestClass = findClosestClass(centroids, flattened_feature_vectors(idx, :))
            red_segmentation_map(i,j) = colors(idx(jdx));
            %idx = idx +1;
            %if closestClass~=1
            %    disp(colors(closestClass));
                %disp([i j]);
                %end
                jdx = jdx+1;
        end
    end
    %segmentation_map(120:122, :) = 255;
    figure;
    image(red_segmentation_map);
    title('3D segmentation');
    figure;
    image(enhanceMap(red_segmentation_map,colors));
    title('3D segmentation after postprocessing');
end
function segMap = enhanceMap(segmentationMap, intensityLevels)
    [h,w] = size(segmentationMap);
    segMap = segmentationMap;
    prev_map = segmentationMap;
    iter_count = 1;
   
    while 1
        segMap = denoiseImage(segMap);
        segMap = dilateImage(segMap, intensityLevels);
        segMap = erodeImage(segMap, intensityLevels);
        if isequal(prev_map, segMap) || iter_count>0
            break;
        end
        iter_count = iter_count + 1;
        prev_map = segMap;
    end
    
end
function out = denoiseImage(img)
    [h, w] = size(img);
    paddedImage = zeros(h+2, w+2);
    out = zeros(h,w);
    for i=2:h
        for j=2:w
            paddedImage(i,j)=img(i-1,j-1);
        end
    end
    for i=2:h
        for j=2:w
            if paddedImage(i-1,j-1)==0 && paddedImage(i-1,j)==0 && paddedImage(i-1,j+1)==0 && paddedImage(i,j-1)==0 && paddedImage(i,j+1)==0 && paddedImage(i+1,j-1)==0 && paddedImage(i+1,j)==0 && paddedImage(i+1,j+1)==0 && paddedImage(i,j)~=0
                out(i-1,j-1) = 0;
            else
                out(i-1,j-1) = img(i-1,j-1);
            end
            
        end
    end
end
function out = erodeImage(img, intensityLevels)
    [h, w] = size(img);
    erodeKernel1 = [0 intensityLevels(1) 0; intensityLevels(1) intensityLevels(1) intensityLevels(1); 0 intensityLevels(1) 0];
    erodeKernel2 = [0 intensityLevels(2) 0; intensityLevels(2) intensityLevels(2) intensityLevels(2); 0 intensityLevels(2) 0];
    erodeKernel3 = [0 intensityLevels(3) 0; intensityLevels(3) intensityLevels(3) intensityLevels(3); 0 intensityLevels(3) 0];
    erodeKernel4 = [0 intensityLevels(4) 0; intensityLevels(4) intensityLevels(4) intensityLevels(4); 0 intensityLevels(4) 0];
    erodeKernel5 = [0 intensityLevels(5) 0; intensityLevels(5) intensityLevels(5) intensityLevels(5); 0 intensityLevels(5) 0];
    paddedImage = zeros(h+2, w+2);
    out = zeros(h,w);
    for i=2:h
        for j=2:w
            paddedImage(i,j)=img(i-1,j-1);
        end
    end
    for i=2:h
        for j=2:w
            if (paddedImage(i-1,j)==intensityLevels(1) && paddedImage(i,j-1)==intensityLevels(1) && paddedImage(i,j+1)==intensityLevels(1) && paddedImage(i+1,j)==intensityLevels(1)) && paddedImage(i,j)==intensityLevels(1)
                out(i-1,j-1) = intensityLevels(1);
            elseif (paddedImage(i-1,j)==intensityLevels(2) && paddedImage(i,j-1)==intensityLevels(2) && paddedImage(i,j+1)==intensityLevels(2) && paddedImage(i+1,j)==intensityLevels(2)) && paddedImage(i,j)==intensityLevels(2)
                out(i-1,j-1) = intensityLevels(2);
            elseif (paddedImage(i-1,j)==intensityLevels(3) && paddedImage(i,j-1)==intensityLevels(3) && paddedImage(i,j+1)==intensityLevels(3) && paddedImage(i+1,j)==intensityLevels(3)) && paddedImage(i,j)==intensityLevels(3)
                out(i-1,j-1) = intensityLevels(3);
            elseif (paddedImage(i-1,j)==intensityLevels(4) && paddedImage(i,j-1)==intensityLevels(4) && paddedImage(i,j+1)==intensityLevels(4) && paddedImage(i+1,j)==intensityLevels(4)) && paddedImage(i,j)==intensityLevels(4)
                out(i-1,j-1) = intensityLevels(4);
            elseif (paddedImage(i-1,j)==intensityLevels(5) && paddedImage(i,j-1)==intensityLevels(5) && paddedImage(i,j+1)==intensityLevels(5) && paddedImage(i+1,j)==intensityLevels(5)) && paddedImage(i,j)==intensityLevels(5)
                out(i-1,j-1) = intensityLevels(5);
            else
                out(i-1, j-1) = 0;
            end
        end
    end
end
function out = dilateImage(img, intensityLevels)
    [h, w] = size(img);
    paddedImage = zeros(h+2, w+2);
    out = zeros(h,w);
    for i=2:h
        for j=2:w
            paddedImage(i,j)=img(i-1,j-1);
        end
    end
    for i=2:h
        for j=2:w
            if(paddedImage(i-1,j)==intensityLevels(1) || paddedImage(i,j-1)==intensityLevels(1) || paddedImage(i,j+1)==intensityLevels(1) || paddedImage(i+1,j)==intensityLevels(1)) && paddedImage(i,j)==intensityLevels(1)
                out(i-1,j-1) = intensityLevels(1);
            elseif(paddedImage(i-1,j)==intensityLevels(2) || paddedImage(i,j-1)==intensityLevels(2) || paddedImage(i,j+1)==intensityLevels(2) || paddedImage(i+1,j)==intensityLevels(2)) && paddedImage(i,j)==intensityLevels(2)
                out(i-1,j-1) = intensityLevels(2);
            elseif(paddedImage(i-1,j)==intensityLevels(3) || paddedImage(i,j-1)==intensityLevels(3) || paddedImage(i,j+1)==intensityLevels(3) || paddedImage(i+1,j)==intensityLevels(3)) && paddedImage(i,j)==intensityLevels(3)
                out(i-1,j-1) = intensityLevels(3);
            elseif(paddedImage(i-1,j)==intensityLevels(4) || paddedImage(i,j-1)==intensityLevels(4) || paddedImage(i,j+1)==intensityLevels(4) || paddedImage(i+1,j)==intensityLevels(4)) && paddedImage(i,j)==intensityLevels(4)
                out(i-1,j-1) = intensityLevels(4);
            elseif(paddedImage(i-1,j)==intensityLevels(5) || paddedImage(i,j-1)==intensityLevels(5) || paddedImage(i,j+1)==intensityLevels(5) || paddedImage(i+1,j)==intensityLevels(5)) && paddedImage(i,j)==intensityLevels(5)
                out(i-1,j-1) = intensityLevels(5);
            else
                out(i-1, j-1) = 0;
            end
        end
    end
end

function closestClass = findClosestClass(centroids, feat_vec)
    min_dist = 1e11;
    class_pred = -1;
    disp(size(feat_vec));
    disp(size(centroids(1,:)));
    dist = euclideanDistance(centroids(1,:), feat_vec);
    if dist<min_dist
        min_dist = dist;
        class_pred = 1;
    end
    dist = euclideanDistance(centroids(2,:), feat_vec);
    if dist<min_dist
        min_dist = dist;
        class_pred = 2;
    end
    dist = euclideanDistance(centroids(3,:), feat_vec);
    if dist<min_dist
        min_dist = dist;
        class_pred = 3;
    end
    dist = euclideanDistance(centroids(4,:), feat_vec);
    if dist<min_dist
        min_dist = dist;
        class_pred = 4;
    end
    dist = euclideanDistance(centroids(5,:), feat_vec);
    if dist<min_dist
        min_dist = dist;
        class_pred = 5;
    end
    closestClass = class_pred;
end
function distance = euclideanDistance(x,y)
    distance = sqrt(sum((x - y).^2,'all'));
end
function res = matmul(a,b)
    [m n] = size(a);
    [p q] = size(b);
    res = zeros(m,q);
    if(n~=p)
        disp("Incorrect dimensions!");
    end
    for i=1:m
        for j=1:n
            for k=1:q
                res(i,k) = res(i,k) + a(i,j) * b(j,k);
            end
        end
    end
end

function filters = generateFilters()
    filters1d = [1 4 6 4 1; -1 -2 0 2 1; -1 0 2 0 -1; -1 2 0 -2 1; 1 -4 6 -4 1];
    l5 = filters1d(1,:);
    e5 = filters1d(2,:);
    s5 = filters1d(3,:);
    w5 = filters1d(4,:);
    r5 = filters1d(5,:);
    filters = zeros(5,5,25);
    filter_num = 1;
    for i=1:5
        for j = 1:5
            l_filter = filters1d(i,:);
            r_filter = filters1d(j,:);
            for t_i=1:5
                for t_j=1:5
                    filters(t_i,t_j,filter_num) = l_filter(t_i)*r_filter(t_j);
                end
            end
            filter_num = filter_num+1;
        end
    end
    
end

function response = convolve5x5(paddedImage, filter, i, j)
    tot = 0;
    tot = tot + paddedImage(i-2,j-2)*filter(1,1);
    tot = tot + paddedImage(i-2,j-1)*filter(1,2);
    tot = tot + paddedImage(i-2,j)*filter(1,3);
    tot = tot + paddedImage(i-2,j+1)*filter(1,4);
    tot = tot + paddedImage(i-2,j+2)*filter(1,5);
    
    tot = tot + paddedImage(i-1,j-2)*filter(2,1);
    tot = tot + paddedImage(i-1,j-1)*filter(2,2);
    tot = tot + paddedImage(i-1,j)*filter(2,3);
    tot = tot + paddedImage(i-1,j+1)*filter(2,4);
    tot = tot + paddedImage(i-1,j+2)*filter(2,5);
    
    tot = tot + paddedImage(i,j-2)*filter(3,1);
    tot = tot + paddedImage(i,j-1)*filter(3,2);
    tot = tot + paddedImage(i,j)*filter(3,3);
    tot = tot + paddedImage(i,j+1)*filter(3,4);
    tot = tot + paddedImage(i,j+2)*filter(3,5);
    
    tot = tot + paddedImage(i+1,j-2)*filter(4,1);
    tot = tot + paddedImage(i+1,j-1)*filter(4,2);
    tot = tot + paddedImage(i+1,j)*filter(4,3);
    tot = tot + paddedImage(i+1,j+1)*filter(4,4);
    tot = tot + paddedImage(i+1,j+2)*filter(4,5);
    
    tot = tot + paddedImage(i+2,j-2)*filter(5,1);
    tot = tot + paddedImage(i+2,j-1)*filter(5,2);
    tot = tot + paddedImage(i+2,j)*filter(5,3);
    tot = tot + paddedImage(i+2,j+1)*filter(5,4);
    tot = tot + paddedImage(i+2,j+2)*filter(5,5);
    
    response = tot;
end
function response = convolve5x5Wrapper(image, filter)
    [h w] = size(image);
    filterResponse = zeros(h, w);
    paddedImage = zeros(h+4, w+4);
    for i=3:h+2
        for j=3:w+2
           paddedImage(i,j) = image(i-2,j-2);
        end
    end
    for i=3:h+2
        for j=3:w+2
            filterResponse(i-2,j-2) = convolve5x5(paddedImage, filter, i,j);
        end
    end
    response = filterResponse;
end
function avgEn = calcAverageEnergyWindowed(response, window_size)
    [h w] = size(response);
    offset = floor(window_size/2);
    avgEn = zeros(h,w);
    paddedImage = zeros(h+offset*2, w+offset*2);
    for i=(1+offset):(h+offset)
        for j=(1+offset):(w+offset)
            paddedImage(i,j) = response(i-offset,j-offset);
        end
    end
    for i=(1+offset):(h+offset)
        for j=(1+offset):(w+offset)
            avgEn(i-offset,j-offset) = calcAvgEn(paddedImage(i-offset:i+offset, j-offset:j+offset));
        end
    end
end
function avgEn = calcAvgEn(window)
    tot = 0;
    [h, w] = size(window);
    for i=1:h
        for j=1:w
           tot = tot + abs(window(i,j)); 
        end
    end
    avgEn = tot/(h*w);
end