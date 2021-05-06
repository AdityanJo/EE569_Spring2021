%
%Adityan Jothi
%USC ID 8162222801
%jothi@usc.edu
%

function problem1
    train_folder = 'dataset_updated/train';
    test_folder = 'dataset_updated/test';
    
    filters = generateFilters();
    train_files = dir([train_folder,filesep,'*.raw']);
    test_files = dir([test_folder,filesep,'*.raw']);
    train_images = zeros(128, 128, length(train_files));
    feature_vectors = zeros(length(train_files), 25);
    for i=1:length(train_files)
        img = single(readraw([train_files(i).folder,filesep,train_files(i).name], 128, 128, 1));
        img = img - mean(img,'all');
        for j=1:25
            response = convolve5x5Wrapper(img, filters(:,:,j));
            avgEn = calcAverageEnergy(response);
            feature_vectors(i,j) = avgEn;
        end
        % Attempt to normalize based on L5L5 response
        %feature_vectors(i,:)=(feature_vectors(i,:))./feature_vectors(i,1);
        
    end
    labels = [1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4];
    discPower = calcDiscriminantPower(feature_vectors, labels, 4);
    figure;
    plot(discPower);
    title("Discriminant Power before PCA");
    disp(discPower);
    
    reduced_feature_vectors_coeff = pca(feature_vectors,"NumComponents",3);
    reduced_feat_vec = matmul(feature_vectors, reduced_feature_vectors_coeff);
    reducedDiscPower= calcDiscriminantPower(reduced_feat_vec, labels, 4);
    figure;
    plot(reducedDiscPower);
    title("Discriminant Power after PCA");
    disp(reducedDiscPower);

    % [36 x 3]  x y z 1-9
    %figure
    %hold on;
    colors = zeros(36, 3);
    for i=1:length(train_files)
        %disp([train_files(i).folder,filesep,train_files(i).name]);
        if labels(i)==1
            colors(i,:) = [1 0 0];
        elseif labels(i)==2
            colors(i,:) = [0 1 0];
        elseif labels(i)==3
            colors(i,:) = [0 0 1];
        else
            colors(i,:) = [0 0 0];
        end
    end
    figure;
    scatter3(reduced_feat_vec(:,1),reduced_feat_vec(:,2), reduced_feat_vec(:,3), 60, colors);
    title("Train set after PCA");
    
    test_feature_vectors = zeros(length(test_files), 25);
    % blanket - 1 brick -2 grass - 3 rice - 4
    test_labels = [3 2 1 3 1 1 2 4 3 2 4 4];
    for i=1:length(test_files)
        img = single(readraw([test_files(i).folder,filesep,test_files(i).name], 128, 128, 1));
        img = img - mean(img,'all');
        for j=1:25
            response = convolve5x5Wrapper(img, filters(:,:,j));
            avgEn = calcAverageEnergy(response);
            test_feature_vectors(i,j) = avgEn;
        end
        %test_feature_vectors(i,:)=(test_feature_vectors(i,:))./test_feature_vectors(i,1); 

    end
    reduced_test_feat_vec = matmul(test_feature_vectors, reduced_feature_vectors_coeff);
    figure;
    scatter3(reduced_test_feat_vec(:,1),reduced_test_feat_vec(:,2), reduced_test_feat_vec(:,3));
    title("Test set after PCA");
    
    runMahNN(feature_vectors, test_feature_vectors, test_files, labels, test_labels);
    runKmeans(feature_vectors, test_feature_vectors, test_files, labels, test_labels);
    runRF(feature_vectors, test_feature_vectors, test_files, labels, test_labels);
    runSVM(feature_vectors, test_feature_vectors, test_files, labels, test_labels);
    %Experimental approach
    runEucNN(feature_vectors, test_feature_vectors, test_files, labels, test_labels);
end
function class_name = getClassName(id)
    if id==1
        class_name = 'BLANKET';
    elseif id==2
        class_name = 'BRICK';
    elseif id==3
        class_name = 'GRASS';
    elseif id==4
        class_name = 'RICE';
    end
end
function drawPreds(test_files, test_labels, test_preds, title_lbl)
    % blanket - 1 brick -2 grass - 3 rice - 4
    images =[];
    grid_init = zeros(length(test_files), 128, 128, 3);
    for i=1:length(test_files)
        img = single(readraw([test_files(i).folder,filesep,test_files(i).name], 128, 128, 1));
        if test_labels(i)==test_preds(i)
            img = insertText(img, [1, 1], [getClassName(test_labels(i)),filesep,getClassName(test_preds(i))],"TextColor","white", "BoxColor","white");
        else
            img = insertText(img, [1, 1], [getClassName(test_labels(i)),filesep,getClassName(test_preds(i))],"TextColor","white", "BoxColor","white");
        end
        grid_init(i,:,:,:) = img;
    end
    form_grid = [squeeze(grid_init(1,:,:,:)), squeeze(grid_init(2,:,:,:)), squeeze(grid_init(3,:,:,:));
        squeeze(grid_init(4,:,:,:)), squeeze(grid_init(5,:,:,:)), squeeze(grid_init(6,:,:,:));
        squeeze(grid_init(7,:,:,:)), squeeze(grid_init(8,:,:,:)), squeeze(grid_init(9,:,:,:));
        squeeze(grid_init(10,:,:,:)), squeeze(grid_init(11,:,:,:)), squeeze(grid_init(12,:,:,:))];
    figure;
    imagesc(uint8(form_grid));
    title(title_lbl);
    
end
function runEucNN(train_feats, test_feats, test_files, train_labels, test_labels)
    for i=1:length(train_feats)
        train_feats(i,:)=(train_feats(i,:))./train_feats(i,1); %-mean(feature_vectors(i,:)))/std(feature_vectors(i,:));
        
    end
    train_feats(:,1) = [];
    for i=1:length(test_files)
        test_feats(i,:)=(test_feats(i,:))./test_feats(i,1); %-mean(feature_vectors(i,:)))/std(feature_vectors(i,:));    
    end
    test_feats(:,1) = [];
    reduced_feature_vectors_coeff = pca(train_feats,"NumComponents",3);
    reduced_feat_vec = matmul(train_feats, reduced_feature_vectors_coeff);
    reduced_test_feat_vec = matmul(test_feats, reduced_feature_vectors_coeff);
    num_corrects = 0;
    preds = [];
    for i=1:length(test_files)
        min_dist = 1e10;
        class_pred = -1;
        dist = computeEucDist(reduced_feat_vec(1:9,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 1;
        end
        dist = computeEucDist(reduced_feat_vec(10:18,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 2;
        end
        dist = computeEucDist(reduced_feat_vec(19:27,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 3;
        end
        dist = computeEucDist(reduced_feat_vec(28:36,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 4;
        end
        preds = [preds class_pred];
        if class_pred == test_labels(i)
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'EucNN');
    disp(['Accuracy [EuclideanNN]:', num2str(num_corrects/length(test_files))]);
end
function runMahNN(train_feats, test_feats, test_files, train_labels, test_labels)
    reduced_feature_vectors_coeff = pca(train_feats,"NumComponents",3);
    reduced_feat_vec = matmul(train_feats, reduced_feature_vectors_coeff);
    reduced_test_feat_vec = matmul(test_feats, reduced_feature_vectors_coeff);
    num_corrects = 0;
    preds=[];
    for i=1:length(test_files)
        min_dist = 1e10;
        class_pred = -1;
        dist = computeMahDist(reduced_feat_vec(1:9,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 1;
        end
        dist = computeMahDist(reduced_feat_vec(10:18,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 2;
        end
        dist = computeMahDist(reduced_feat_vec(19:27,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 3;
        end
        dist = computeMahDist(reduced_feat_vec(28:36,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 4;
        end
        preds = [preds class_pred];
        if class_pred == test_labels(i)
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'MahNN');
    disp(['Accuracy [MahNN]:', num2str(num_corrects/length(test_files))]);
end
function runKmeans(train_feats, test_feats, test_files, train_labels, test_labels)
    %for i=1:length(train_feats)
    %    train_feats(i,:)=(train_feats(i,:))./train_feats(i,1);
        
    %end
    %train_feats(:,1) = [];
    
    reduced_feature_vectors_coeff = pca(train_feats,"NumComponents",3);
    reduced_feat_vec = matmul(train_feats, reduced_feature_vectors_coeff);
    reduced_test_feat_vec = matmul(test_feats, reduced_feature_vectors_coeff);
    
    [idx, centroids] = kmeans(reduced_feat_vec, 4);
    
    centroid_map = [0 0 0 0];
    min_dist = euclideanDistance(centroids(1,:), reduced_feat_vec(1));
    min_idx = 1;    
    for i=2:36
        dist = euclideanDistance(centroids(1,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(1) = train_labels(min_idx);
    
    min_dist = 1e6;
    min_idx = -1;    
    for i=1:36
        if any(centroid_map(:)==train_labels(i))
            continue;
        end
        dist = euclideanDistance(centroids(2,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(2) = train_labels(min_idx);
    min_dist = 1e10;
    min_idx = -1;    
    for i=1:36
        if any(centroid_map(:)==train_labels(i))
            continue;
        end
        dist = euclideanDistance(centroids(3,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(3) = train_labels(min_idx);
    centroid_map(centroid_map==0) =4;
    
    %reassign centroid numbers to class ids 
    
    num_corrects = 0;
    preds=[];
    for i=1:length(test_files)
        min_dist = 1e6;
        class_pred = -1;
        dist = euclideanDistance(centroids(1,:), reduced_test_feat_vec(i));
        %disp(dist);
        if dist < min_dist
            min_dist = dist;
            class_pred = 1;
        end
        dist = euclideanDistance(centroids(2,:), reduced_test_feat_vec(i));
        %disp(dist);
        if dist < min_dist
            min_dist = dist;
            class_pred = 2;
        end
        dist = euclideanDistance(centroids(3,:), reduced_test_feat_vec(i));
        %disp(dist);
        if dist < min_dist
            min_dist = dist;
            class_pred = 3;
        end
        dist = euclideanDistance(centroids(4,:), reduced_test_feat_vec(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 4;
        end
        class_pred = centroid_map(class_pred);
        preds = [preds class_pred];
        if class_pred == test_labels(i)
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'KMeans 3D');
    disp(['Accuracy [KMeans 3-D]:', num2str(num_corrects/length(test_files))]);
    
    [idx, centroids] = kmeans(train_feats, 4);
    centroid_map = [0 0 0 0];
    min_dist = euclideanDistance(centroids(1,:), reduced_feat_vec(1));
    min_idx = 1;    
    for i=2:36
        dist = euclideanDistance(centroids(1,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(1) = train_labels(min_idx);
    
    min_dist = 1e10;
    min_idx = -1;    
    for i=1:36
        if any(centroid_map(:)==train_labels(i))
            continue;
        end
        dist = euclideanDistance(centroids(2,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(2) = train_labels(min_idx);
    min_dist = 1e6;
    min_idx = -1;    
    for i=1:36
        if any(centroid_map(:)==train_labels(i))
            continue;
        end
        dist = euclideanDistance(centroids(3,:), reduced_feat_vec(i));
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
    end
    centroid_map(3) = train_labels(min_idx);
    centroid_map(centroid_map==0) =4;
    num_corrects = 0;
    preds =[];
    for i=1:length(test_files)
        min_dist = 1e10;
        class_pred = -1;
        dist = euclideanDistance(centroids(1,:), test_feats(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 1;
        end
        dist = euclideanDistance(centroids(2,:), test_feats(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 2;
        end
        dist = euclideanDistance(centroids(3,:), test_feats(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 3;
        end
        dist = euclideanDistance(centroids(4,:), test_feats(i));
        if dist < min_dist
            min_dist = dist;
            class_pred = 4;
        end
        class_pred = centroid_map(class_pred);
        preds = [preds class_pred];
        if class_pred == test_labels(i)
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'KMeans 25D');
    disp(['Accuracy [KMeans 25-D]:', num2str(num_corrects/length(test_files))]);
end
function runRF(train_feats, test_feats, test_files, train_labels, test_labels)
    reduced_feature_vectors_coeff = pca(train_feats,"NumComponents",3);
    reduced_feat_vec = matmul(train_feats, reduced_feature_vectors_coeff);
    reduced_test_feat_vec = matmul(test_feats, reduced_feature_vectors_coeff);
    
    Mdl = TreeBagger(100,reduced_feat_vec,train_labels,'OOBPrediction','On',...
    'Method','classification');
    [y_pred] = predict(Mdl, reduced_test_feat_vec);
    num_corrects = 0;
    y_pred = str2num((cell2mat(y_pred)));
    preds = [];
    for i=1:length(test_files)
        preds = [preds double(y_pred(i))];
        if(double(y_pred(i)) == double(test_labels(i)))
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'Random Forest');
    disp(['Accuracy [Random Forest]:', num2str(num_corrects/length(test_files))]);
end
function runSVM(train_feats, test_feats, test_files, train_labels, test_labels)
    reduced_feature_vectors_coeff = pca(train_feats,"NumComponents",3);
    reduced_feat_vec = matmul(train_feats, reduced_feature_vectors_coeff);
    reduced_test_feat_vec = matmul(test_feats, reduced_feature_vectors_coeff);
    
    Mdl = fitcecoc(reduced_feat_vec, train_labels);
    [y_pred] = predict(Mdl, reduced_test_feat_vec);
    num_corrects = 0;
    y_pred = y_pred';
    preds =[];
    for i=1:length(test_files)
        preds = [preds double(y_pred(i))];
        if(double(y_pred(i)) == double(test_labels(i)))
            num_corrects = num_corrects + 1;
        end
    end
    drawPreds(test_files, test_labels, preds, 'SVM');
    disp(['Accuracy [SVM]:', num2str(num_corrects/length(test_files))]);
end
function matrix = computeCovMatrix(train_vec)
    [n_samples, feat_size] = size(train_vec);
    mean_vec = mean(train_vec, 1);
    matrix = zeros(feat_size, feat_size);
    for i=1:n_samples
        l_vec = train_vec(i,:) - mean_vec;
        mat = matmul(l_vec', l_vec);
        matrix = matrix + mat;
    end
    matrix = matrix/(n_samples-1);
end
function distance = computeMahDist(train_vec, test_vec)
    S = computeCovMatrix(train_vec);
    mean_vec = mean(train_vec,1);
    distance = sqrt(matmul(matmul((test_vec-mean_vec),inv(S)),(test_vec-mean_vec)'));
end
function distance = euclideanDistance(x,y)
    distance = sqrt(sum((x - y).^2,'all'));
end
function distance = computeEucDist(train_vec, test_vec)
    min_dist = 10000;
    for i=1:length(train_vec)
        dist = sqrt(sum((train_vec(i, :) - test_vec).^2,'all'));
        if dist<min_dist
            min_dist = dist;
        end
    end
    distance = min_dist;
end
function discPower = calcDiscriminantPower(feature_vectors, labels, n_classes)
    [n_samples, feat_size] = size(feature_vectors);
    per_feat_avg = mean(feature_vectors, 1);
    per_class_feat_avg = [mean(feature_vectors(1:9,:),1); mean(feature_vectors(10:18,:),1); mean(feature_vectors(19:27,:),1); mean(feature_vectors(28:36,:),1)];
    intra_class_ss = zeros(1,feat_size);
    inter_class_ss = zeros(1,feat_size);
    
    for i=1:feat_size
        for j=1:n_samples
            intra_class_ss(1,i) = intra_class_ss(1,i) + (feature_vectors(j,i) - per_class_feat_avg(labels(j),i))^2;
            inter_class_ss(1,i) = inter_class_ss(1,i) + (per_feat_avg(1,i) - per_class_feat_avg(labels(j),i))^2;
        end
    end
    discPower = intra_class_ss./inter_class_ss;
end
function avgEn = calcAverageEnergy(response)
    [h w] = size(response);
    tot = 0;
    for i=1:h
        for j=1:w
            tot = tot + abs(response(i,j));
        end
    end
    avgEn = tot/(h*w);
end
function response = convolve5x5Wrapper(image, filter)
    filterResponse = zeros(128, 128);
    paddedImage = zeros(132, 132);
    for i=3:130
        for j=3:130
           paddedImage(i,j) = image(i-2,j-2);
        end
    end
    for i=3:130
        for j=3:130
            filterResponse(i-2,j-2) = convolve5x5(paddedImage, filter, i,j);
        end
    end
    response = filterResponse;
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