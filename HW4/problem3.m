%
%Adityan Jothi
%USC ID 8162222801
%jothi@usc.edu
%

function problem3()
    dog_1_rgb = (imread("Dog_1.png"));
    dog_2_rgb = (imread("Dog_2.png"));
    dog_3_rgb = (imread("Dog_3.png"));
    cat_rgb= (imread("Cat.png"));
   
    dog_1 = rgb2gray(dog_1_rgb);
    dog_2 = rgb2gray(dog_2_rgb);
    dog_3 = rgb2gray(dog_3_rgb);
    cat = rgb2gray(cat_rgb);
   
    [dog_1_fts, dog_1_dsc] = vl_sift(single(dog_1));
    [dog_2_fts, dog_2_dsc] = vl_sift(single(dog_2));
    [dog_3_fts, dog_3_dsc] = vl_sift(single(dog_3));
    [cat_fts, cat_dsc] = vl_sift(single(cat));
    
    matchLSS(dog_3_rgb, dog_1_rgb, dog_3_fts, dog_1_fts, dog_3_dsc, dog_1_dsc);
    matchImages(dog_3_rgb, dog_1_rgb, dog_3_fts, dog_1_fts, dog_3_dsc, dog_1_dsc, 1.7, "Dog_1 & Dog_3");
    matchImages(dog_3_rgb, dog_2_rgb, dog_3_fts, dog_2_fts, dog_3_dsc, dog_2_dsc, 1.8, "Dog_3 & Dog_2");
    matchImages(dog_3_rgb, cat_rgb, dog_3_fts, cat_fts, dog_3_dsc, cat_dsc,1.75, "Dog_3 & Cat");
    matchImages(dog_1_rgb, cat_rgb, dog_1_fts, cat_fts, dog_1_dsc, cat_dsc,1.6, "Dog_1 & Cat");
    
    doBagOfWords(["Dog_1.png", "Dog_2.png", "Dog_3.png", "Cat.png"], dog_3);
end
function matchImages(img_left, img_right, img_left_fts, img_right_fts, img_left_dsc, img_right_dsc, threshold, title_lbl)
    [matches, scores] = vl_ubcmatch(img_left_dsc, img_right_dsc, threshold);
    [t num_matches]=size(matches);
    selected_fts_left = [];
    selected_fts_right = [];
    
    for i=1:num_matches 
        selected_fts_left = [selected_fts_left img_left_fts(:,matches(1,i))];
        selected_fts_right = [selected_fts_right img_right_fts(:,matches(2,i))];
    end
    figure;
    imagesc([img_left, img_right]);
    title(title_lbl);
    selected_fts_right(1,:) = selected_fts_right(1,:) + 640;
    vl_plotframe(selected_fts_left);
    vl_plotframe(selected_fts_right);
    for i=1:num_matches
        line([selected_fts_left(1,i) selected_fts_right(1,i)], [[selected_fts_left(2,i) selected_fts_right(2,i)]]);
    end
end
function matchLSS(img_left, img_right, img_left_fts, img_right_fts, img_left_dsc, img_right_dsc)
    [ft_shape, num_points]=size(img_left_fts);
    max_scale = img_left_fts(3,1);
    max_scale_idx = 1;
    for i=2:num_points
        if img_left_fts(3,i)>max_scale
            max_scale = img_left_fts(3,i);
            max_scale_idx = i;
        end
    end
    disp(['Orientation L:', num2str(img_left_fts(4,max_scale_idx))]);
    [dsc_shape, num_points_r] = size(img_right_dsc);
    min_dst = 1e6;
    
    match_idx = -1;
    distances = zeros(num_points_r);
    for i=1:num_points_r;
        dist = euclideanDistance(img_left_dsc(:, max_scale_idx), img_right_dsc(:,i));
        distances(i) = dist;
        if(dist<min_dst)
            min_dst = dist;
            match_idx = i;
        end
    end
    disp(['Orientation R:', num2str(img_right_fts(4,match_idx))]);
    figure;
    imshow(img_left);
    title("Left Image (Dog_3)");
    h1 = vl_plotframe(img_left_fts(:,max_scale_idx));
    h3 = vl_plotsiftdescriptor(img_left_dsc(:,max_scale_idx), img_left_fts(:,max_scale_idx));
    
    set(h1,'color','k','linewidth',3) ;
    set(h3,'color','g','linewidth',3) ;
    %match_idx = 876;
    figure;
    image(uint8(img_right));
    title("Right Image (Dog_1)");
    h1 = vl_plotframe(img_right_fts(:,match_idx));
    h3 = vl_plotsiftdescriptor(img_right_dsc(:,match_idx), img_right_fts(:,match_idx));
    
    set(h1,'color','k','linewidth',3) ;
    set(h3,'color','g','linewidth',3) ;
    figure;
    plot(1:num_points_r, distances);
    title('Distance to descriptors (NNSearch)');
    
end
function [similarity] = doBagOfWords(images, search_image)
    num_files = length(images);
    dscs = [];
    for i=1:num_files
        [fts, dsc] = vl_sift(single(rgb2gray(imread(images(i)))));
        dsc = single(dsc');
        dsc_coeff = pca(single(dsc),"NumComponents",20);
        dsc_ft_vec = matmul(single(dsc), dsc_coeff);
        dscs = [dscs; dsc_ft_vec];
    end
    [idx, centroids] = kmeans(dscs, 8);
    histogramArr = zeros(4,8);
    for i=1:num_files
        [fts, dsc] = vl_sift(single(rgb2gray(imread(images(i)))));
        dsc = single(dsc');
        dsc_coeff = pca(single(dsc),"NumComponents",20);
        dsc_ft_vec = matmul(single(dsc), dsc_coeff);
        [num_feats, ft_shape] = size(dsc_ft_vec);
        for j=1:num_feats
            k = findClosestCentroid(centroids, dsc_ft_vec(j,:));
            histogramArr(i,k) = histogramArr(i,k)+1;
        end
    end
    
    disp(histogramArr);
    figure;
    bar(histogramArr(1,:)');
    title('BoW Histogram Dog_1');
    figure;
    bar(histogramArr(2,:)');
    title('BoW Histogram Dog_2');
    figure;
    bar(histogramArr(3,:)');
    title('BoW Histogram Dog_3');
    figure;
    bar(histogramArr(4,:)');
    title('BoW Histogram Cat');
    scores = [compareHistogram(histogramArr(1,:), histogramArr(3,:)),
        compareHistogram(histogramArr(2,:), histogramArr(3,:)),
        compareHistogram(histogramArr(4,:), histogramArr(3,:))];
    figure();
    bar(scores);
    title('Similarity Scores for Dog_3');
    xlabel(['Dog_1', 'Dog_2', 'Cat']);
    disp(compareHistogram(histogramArr(1,:), histogramArr(3,:)));
    disp(compareHistogram(histogramArr(2,:), histogramArr(3,:)));
    disp(compareHistogram(histogramArr(4,:), histogramArr(3,:)));
end
function histValue = compareHistogram(l_vec, r_vec)
    numer = 0;
    denom = 0;
    for i=1:8;
        min_ = min(l_vec(i), r_vec(i));
        max_ = max(l_vec(i), r_vec(i));
        numer = numer+min_;
        denom = denom+max_;
    end
    histValue = numer/denom;
end
function idx = findClosestCentroid(centroids, feat)
    
    min_dist = euclideanDistance(centroids(1,:), feat);
    min_idx = 1;
    distances = [min_dist];
    for i=2:8
        dist = euclideanDistance(centroids(i,:), feat);
        if dist<min_dist
            min_dist = dist;
            min_idx = i;
        end
        distances = [distances dist];
    end
    idx = min_idx;
end
function distance = euclideanDistance(x,y)
    distance = sqrt(sum((x - y).^2,'all'));
end