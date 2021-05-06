/*
  Adityan Jothi
  USC ID 8162222801
  jothi@usc.edu  
*/

#include "common.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class ProblemTwoSolver{
    public:
        void invertMatrix(float matrix[3][3], float matrixInv[3][3]){
            float determinant = matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1]) - matrix[0][1] * (matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0]) + matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]);
            // cout << "Determinant : " << determinant << endl;
            if (determinant==0){
                cout << "Non-invertible matrix with determinant:" << determinant << endl;
                return;
            }
            float res[3][3];

            res[0][0] = matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1];
            res[0][1] = -1*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0]);
            res[0][2] = matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0];
            res[1][0] = -1*(matrix[0][1]*matrix[2][2]-matrix[0][2]*matrix[2][1]);
            res[1][1] = matrix[0][0]*matrix[2][2]-matrix[0][2]*matrix[2][0];
            res[1][2] = -1*(matrix[0][0]*matrix[2][1]-matrix[0][1]*matrix[2][0]);
            res[2][0] = matrix[0][1]*matrix[1][2]-matrix[0][2]*matrix[1][1];
            res[2][1] = -1*(matrix[0][0]*matrix[1][2]-matrix[0][2]*matrix[1][0]);
            res[2][2] = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];

            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    matrixInv[j][i] = (float)res[i][j]/determinant;
                }
            }

        }
        void stitchThree(char *leftFileLocation, char* midFileLocation, char *rightFileLocation, int leftWidth, int leftHeight, int midWidth, int midHeight, int rightWidth, int rightHeight, int bytesPerPixel, char *outFileLocation, float H_left[3][3], float H_right[3][3]){
            Image leftImg(leftFileLocation, leftWidth, leftHeight, bytesPerPixel);
            Image midImg(midFileLocation, midWidth, midHeight, bytesPerPixel);
            Image rightImg(rightFileLocation, rightWidth, rightHeight, bytesPerPixel);

            int max_width = leftWidth+midWidth+rightWidth;
            int max_height = (leftHeight>midHeight)?((leftHeight>rightHeight)?leftHeight:rightHeight):midHeight>rightHeight?midHeight:rightHeight;

            cout << "Result in " << max_width << "x" << max_height << endl;

            Image res(max_width, max_height, bytesPerPixel);
            for(int i=0;i<midHeight;i++){
                for(int j=0;j<midWidth;j++){
                    res.set(midImg.get(i,j,0),i,leftWidth+j,0);
                    res.set(midImg.get(i,j,1),i,leftWidth+j,1);
                    res.set(midImg.get(i,j,2),i,leftWidth+j,2);
                }
            }

            float H_inv_left[3][3]={{0,0,0},{0,0,0},{0,0,0}};
            float H_inv_right[3][3]={{0,0,0},{0,0,0},{0,0,0}};
            invertMatrix(H_left,H_inv_left);
            invertMatrix(H_right,H_inv_right);

            for(int i=0; i<max_height; i++){
                for(int j=0; j<max_width; j++){
                    float x = j-0.5;
                    float y = max_height - i + 0.5;
                    float xl_ = H_inv_left[0][0]*x+H_inv_left[0][1]*y+H_inv_left[0][2];
                    float yl_ = H_inv_left[1][0]*x+H_inv_left[1][1]*y+H_inv_left[1][2];
                    float wl_ = H_inv_left[2][0]*x+H_inv_left[2][1]*y+H_inv_left[2][2];
                    xl_/=wl_;
                    yl_/=wl_;

                    float jl_ = xl_+0.5;
                    float il_ = -(yl_-0.5-leftHeight);

                    float il_i = floor(il_); //For interpolation
                    float jl_j = floor(jl_);

                    float xr_ = H_inv_right[0][0]*x+H_inv_right[0][1]*y+H_inv_right[0][2];
                    float yr_ = H_inv_right[1][0]*x+H_inv_right[1][1]*y+H_inv_right[1][2];
                    float wr_ = H_inv_right[2][0]*x+H_inv_right[2][1]*y+H_inv_right[2][2];
                    xr_/=wr_;
                    yr_/=wr_;

                    float jr_ = xr_+0.5;
                    float ir_ = -(yr_-0.5-rightHeight);

                    float ir_i = floor(ir_); //For interpolation
                    float jr_j = floor(jr_);
                    

                    if(j<max_width/2){
                        res.set(leftImg.get(il_i,jl_j,0,false),i,j,0);
                        res.set(leftImg.get(il_i,jl_j,1,false),i,j,1);
                        res.set(leftImg.get(il_i,jl_j,2,false),i,j,2);

                    }
                    else                    
                    {
                        res.set(rightImg.get(ir_i,jr_j,0,false),i,j,0);
                        res.set(rightImg.get(ir_i,jr_j,1,false),i,j,1);
                        res.set(rightImg.get(ir_i,jr_j,2,false),i,j,2);
                    
                    }
                    

                        
                    
                    
                }
            }

            
            res.saveImage(outFileLocation);
        }
        void stitch(char *leftFileLocation, char *rightFileLocation, int leftWidth, int leftHeight, int rightWidth, int rightHeight, int bytesPerPixel, char* outFileLocation, float H[3][3]){
            Image leftImg(leftFileLocation, leftWidth, leftHeight, bytesPerPixel);
            Image rightImg(rightFileLocation, rightWidth, rightHeight, bytesPerPixel);

            Ptr<SURF> detector = SURF::create(400, 4, 3, true, true);
            vector<KeyPoint> keypointsLeft, keypointsRight;
            Mat leftDescriptors, rightDescriptors;
            
            unsigned char leftData[leftHeight][leftWidth][bytesPerPixel];
            unsigned char rightData[rightHeight][rightWidth][bytesPerPixel];

            for(int i=0; i<leftHeight; i++){
                for(int j=0;j<leftWidth; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        leftData[i][j][k] = leftImg.get(i,j,k);
                    }
                }
            }
            for(int i=0; i<rightHeight; i++){
                for(int j=0;j<rightWidth; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        rightData[i][j][k] = rightImg.get(i,j,k);
                    }
                }
            }

            Mat leftImage(leftHeight, leftWidth, CV_8U, leftData);
            Mat rightImage(rightHeight, rightWidth, CV_8U, rightData);

            detector->detectAndCompute(leftImage, noArray(), keypointsLeft, leftDescriptors);
            detector->detectAndCompute(rightImage, noArray(), keypointsRight, rightDescriptors);

            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            vector<vector<DMatch>> knn_matches;
            matcher->knnMatch(leftDescriptors, rightDescriptors, knn_matches, 2);

            const float ratio_thresh=0.7f;
            vector<DMatch> good_matches;

            float closest_top_left[2][2];
            float closest_top_right[2][2];
            float closest_bottom_left[2][2];
            float closest_bottom_right[2][2];

            closest_top_left[0][0] = keypointsLeft[knn_matches[0][0].queryIdx].pt.x;
            closest_top_left[1][0] = keypointsRight[knn_matches[0][0].queryIdx].pt.x;
            closest_top_left[0][1] = keypointsLeft[knn_matches[0][0].queryIdx].pt.y;
            closest_top_left[1][1] = keypointsRight[knn_matches[0][0].queryIdx].pt.y;

            closest_top_right[0][0] = keypointsLeft[knn_matches[0][0].queryIdx].pt.x;
            closest_top_right[1][0] = keypointsRight[knn_matches[0][0].queryIdx].pt.x;
            closest_top_right[0][1] = keypointsLeft[knn_matches[0][0].queryIdx].pt.y;
            closest_top_right[1][1] = keypointsRight[knn_matches[0][0].queryIdx].pt.y;
            
            closest_bottom_left[0][0] = keypointsLeft[knn_matches[0][0].queryIdx].pt.x;
            closest_bottom_left[1][0] = keypointsRight[knn_matches[0][0].queryIdx].pt.x;
            closest_bottom_left[0][1] = keypointsLeft[knn_matches[0][0].queryIdx].pt.y;
            closest_bottom_left[1][1] = keypointsRight[knn_matches[0][0].queryIdx].pt.y;
            
            closest_bottom_right[0][0] = keypointsLeft[knn_matches[0][0].queryIdx].pt.x;
            closest_bottom_right[1][0] = keypointsRight[knn_matches[0][0].queryIdx].pt.x;
            closest_bottom_right[0][1] = keypointsLeft[knn_matches[0][0].queryIdx].pt.y;
            closest_bottom_right[1][1] = keypointsRight[knn_matches[0][0].queryIdx].pt.y;
            
            for(int i=0; i<knn_matches.size(); i++){
                if(knn_matches[i][0].distance < ratio_thresh*knn_matches[i][i].distance){
                    good_matches.push_back(knn_matches[i][0]);
                    
                    float left_x = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                    float left_y = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    
                    float right_x = keypointsRight[knn_matches[i][0].queryIdx].pt.x;
                    float right_y = keypointsRight[knn_matches[i][0].queryIdx].pt.y;

                    if((left_x*left_x+left_y*left_y)<(closest_top_left[0][0]*closest_top_left[0][0]+closest_top_left[0][1]*closest_top_left[0][1])){
                        closest_top_left[0][0] = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                        closest_top_left[0][1] = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((leftWidth-left_x)*(leftWidth-left_x)+left_y*left_y)<((leftWidth-closest_top_right[0][0])*(leftWidth-closest_top_right[0][0])+closest_top_right[0][1]*closest_top_right[0][1])){
                        closest_top_right[0][0] = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                        closest_top_right[0][1] = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((leftWidth-left_x)*(leftWidth-left_x)+(leftHeight-left_y)*(leftHeight-left_y))<((leftWidth-closest_bottom_right[0][0])*(leftWidth-closest_bottom_right[0][0])+(leftHeight-closest_bottom_right[0][1])*(leftHeight-closest_bottom_right[0][1]))){
                        closest_bottom_right[0][0] = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                        closest_bottom_right[0][1] = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((left_x)*(left_x)+(leftHeight-left_y)*(leftHeight-left_y))<((closest_bottom_right[0][0])*(closest_bottom_right[0][0])+(leftHeight-closest_bottom_right[0][1])*(leftHeight-closest_bottom_right[0][1]))){
                        closest_bottom_left[0][0] = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                        closest_bottom_left[0][1] = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    }
                    
                    if((right_x*right_x+right_y*right_y)<(closest_top_left[1][0]*closest_top_left[1][0]+closest_top_left[1][1]*closest_top_left[1][1])){
                        closest_top_left[1][0] = keypointsRight[knn_matches[i][0].queryIdx].pt.x;
                        closest_top_left[1][1] = keypointsRight[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((rightWidth-right_x)*(rightWidth-right_x)+right_y*right_y)<((rightWidth-closest_top_right[1][0])*(rightWidth-closest_top_right[1][0])+closest_top_right[1][1]*closest_top_right[1][1])){
                        closest_top_right[1][0] = keypointsRight[knn_matches[i][0].queryIdx].pt.x;
                        closest_top_right[1][1] = keypointsRight[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((rightWidth-right_x)*(rightWidth-right_x)+(rightHeight-right_y)*(rightHeight-right_y))<((rightWidth-closest_bottom_right[1][0])*(rightWidth-closest_bottom_right[1][0])+(rightHeight-closest_bottom_right[1][1])*(rightHeight-closest_bottom_right[1][1]))){
                        closest_bottom_right[1][0] = keypointsRight[knn_matches[i][0].queryIdx].pt.x;
                        closest_bottom_right[1][1] = keypointsRight[knn_matches[i][0].queryIdx].pt.y;
                    }
                    if(((right_x)*(right_x)+(rightHeight-right_y)*(rightHeight-right_y))<((closest_bottom_right[1][0])*(closest_bottom_right[1][0])+(rightHeight-closest_bottom_right[1][1])*(rightHeight-closest_bottom_right[1][1]))){
                        closest_bottom_left[1][0] = keypointsLeft[knn_matches[i][0].queryIdx].pt.x;
                        closest_bottom_left[1][1] = keypointsLeft[knn_matches[i][0].queryIdx].pt.y;
                    }

                    
                    // cout << keypointsLeft[knn_matches[i][0].queryIdx].pt.x << " " << keypointsLeft[knn_matches[i][0].queryIdx].pt.y << "->";
                    // cout << keypointsRight[knn_matches[i][0].queryIdx].pt.x << " " << keypointsRight[knn_matches[i][0].queryIdx].pt.y << endl;
                }
            }

            cout << closest_top_left[0][0] << "," << closest_top_left[0][1] << endl;
            cout << closest_top_right[0][0] << "," << closest_top_right[0][1] << endl;
            cout << closest_bottom_left[0][0] << "," << closest_bottom_left[0][1] << endl;
            cout << closest_bottom_right[0][0] << "," << closest_bottom_right[0][1] << endl; 

            cout << endl;


            cout << closest_top_left[1][0] << "," << closest_top_left[1][1] << endl;
            cout << closest_top_right[1][0] << "," << closest_top_right[1][1] << endl;
            cout << closest_bottom_left[1][0] << "," << closest_bottom_left[1][1] << endl;
            cout << closest_bottom_right[1][0] << "," << closest_bottom_right[1][1] << endl; 

            int maxWidth = leftWidth+rightWidth;
            int maxHeight = (leftHeight>rightHeight)?leftHeight:rightHeight;
            cout << leftWidth << "x" << rightWidth << endl;
            cout << "Result in " << maxWidth << "x" << maxHeight << endl;
            Image res(maxWidth, maxHeight, 3);

            for(int i=0;i<rightHeight;i++){
                for(int j=0;j<rightWidth;j++){
                    res.set(rightImg.get(i,j,0),i,leftWidth+j,0);
                    res.set(rightImg.get(i,j,1),i,leftWidth+j,1);
                    res.set(rightImg.get(i,j,2),i,leftWidth+j,2);
                }
            }
            res.saveImage("inter.raw");
            float H_inv[3][3]={{0,0,0},{0,0,0},{0,0,0}};
            invertMatrix(H,H_inv);
            for(int i=0; i<maxHeight; i++){
                for(int j=0; j<maxWidth; j++){
                    float x = j-0.5;
                    float y = maxHeight - i + 0.5;
                    float x_ = H_inv[0][0]*x+H_inv[0][1]*y+H_inv[0][2];
                    float y_ = H_inv[1][0]*x+H_inv[1][1]*y+H_inv[1][2];
                    float w_ = H_inv[2][0]*x+H_inv[2][1]*y+H_inv[2][2];
                    x_/=w_;
                    y_/=w_;

                    float j_ = x_+0.5;
                    float i_ = -(y_-0.5-leftHeight);

                    float i_i = floor(i_); //For interpolation
                    float j_j = floor(j_);
                    
                    if(i_>leftImg.height || j_>leftImg.width || i_<0 || j_<0)
                        continue;
                    
                    res.set(leftImg.get(i_i,j_j,0,false),i,j,0);
                    res.set(leftImg.get(i_i,j_j,1,false),i,j,1);
                    res.set(leftImg.get(i_i,j_j,2,false),i,j,2);
                    
                }
            }
            res.saveImage(outFileLocation);
        
            // for(int i=0;i<leftHeight; i++){
            //     for(int j=0; j<leftWidth; j++){
            //         float x = j-0.5;
            //         float y = leftHeight - i + 0.5;

            //         float x_ = H[0][0]*x+H[0][1]*y+H[0][2];
            //         float y_ = H[1][0]*x+H[1][1]*y+H[1][2];
            //         float w_ = H[2][0]*x+H[2][1]*y+H[2][2];

            //         x_*=w_;
            //         y_*=w_;

            //         float j_ = x_+0.5;
            //         float i_ = -(y_-0.5-leftHeight);

            //         float i_i = floor(i_); //For interpolation
            //         float j_i = floor(j_);


            //         // cout << i_ << " " << j_ << endl;

            //         // res.set((leftImg.get(i_i+1,j_i+1,0)+leftImg.get(i_i,j_i+1,0)+leftImg.get(i_i+1,j_i,0)+leftImg.get(i_i,j_i,0))/4,i_,j_,0);
            //         // res.set((leftImg.get(i_i+1,j_i+1,1)+leftImg.get(i_i,j_i+1,1)+leftImg.get(i_i+1,j_i,1)+leftImg.get(i_i,j_i,1))/4,i_,j_,1);
            //         // res.set((leftImg.get(i_i+1,j_i+1,2)+leftImg.get(i_i,j_i+1,2)+leftImg.get(i_i+1,j_i,2)+leftImg.get(i_i,j_i,2))/4,i_,j_,2);

            //         res.set(leftImg.get(i,j,0),i_,j_,0);
            //         res.set(leftImg.get(i,j,1),i_,j_,1);
            //         res.set(leftImg.get(i,j,2),i_,j_,2);
            //     }
            // }
            cout << outFileLocation << endl;
            
            

        }
        void performStitch(char *leftFileLocation, char *midFileLocation, char *rightFileLocation, int leftWidth, int leftHeight, int midWidth, int midHeight, int rightWidth, int rightHeight, int bytesPerPixel, char *outFileLocation){
            // float H_left_mid[3][3]={
            //     {0.9878, 0.6014, 0.0028},
            //     {-1.7938, -1.0176, -0.0058},
            //     {286.6289, 168.2758, 1.0}
            // };
            float H_left_mid[3][3]={
                {0.400417, -0.02428, 166.6544983},
                {-0.40359, 0.8537, 33.239398},
                {-0.00185839, 0.00007308, 1.0}
            };
            stitch(leftFileLocation, midFileLocation, leftWidth, leftHeight, midWidth, midHeight, bytesPerPixel, "left_mid.raw", H_left_mid);

            float H_mid_right[3][3]={
                {0.024435, -0.039631, 278.37038},
                {-0.660529, 0.91866651, 18.9158708},
                {-0.00295, 0.00006, 1.0}
            };
            // float H_left_mid_right[3][3]={
            //     {-7.7826, -0.26695, 377.023977},
            //     {-1.4346, -0.06608, 64.78314},
            //     {-0.02154, -0.00005, 1.0}
            // };

            
            stitch(midFileLocation, rightFileLocation, midWidth, midHeight, rightWidth, rightHeight, bytesPerPixel, "mid_right.raw", H_mid_right);
            
            float H_left_mid_right[3][3]={
                {0.83604, -0.007, 50.7203},
                {-0.10370, 0.97098, -5.16456},
                {-0.0005843, 0.00012614, 1}
            };
            

            // stitch("left_mid.raw", "mid_right.raw", leftWidth+midWidth, leftHeight>midHeight?leftHeight:midHeight, midWidth+rightWidth, midHeight>rightHeight?midHeight:rightHeight, bytesPerPixel, outFileLocation, H_mid_right);

            stitchThree(leftFileLocation, midFileLocation, rightFileLocation, leftWidth, leftHeight, midWidth, midHeight, rightWidth, rightHeight, bytesPerPixel, outFileLocation, H_left_mid, H_mid_right);
            
            // Image leftImg(leftFileLocation, leftWidth, leftHeight, bytesPerPixel);
            // Image midImg(midFileLocation, leftWidth, leftHeight, bytesPerPixel);
            // Image rightImg(rightFileLocation, leftWidth, leftHeight, bytesPerPixel);

            // Ptr<SURF> detector = SURF::create(400, 4, 3, true, true);
            // vector<KeyPoint> leftKeypoints, midKeypoints, rightKeypoints;

            // unsigned char leftData[leftHeight][leftWidth][bytesPerPixel];
            // unsigned char midData[midHeight][midWidth][bytesPerPixel];
            // unsigned char rightData[rightHeight][rightWidth][bytesPerPixel];

            // for(int i=0; i<leftHeight; i++){
            //     for(int j=0;j<leftWidth; j++){
            //         for(int k=0; k<bytesPerPixel; k++){
            //             leftData[i][j][k] = leftImg.get(i,j,k);
            //         }
            //     }
            // }
            // for(int i=0; i<midHeight; i++){
            //     for(int j=0;j<midWidth; j++){
            //         for(int k=0; k<bytesPerPixel; k++){
            //             midData[i][j][k] = midImg.get(i,j,k);
            //         }
            //     }
            // }
            // for(int i=0; i<rightHeight; i++){
            //     for(int j=0;j<rightWidth; j++){
            //         for(int k=0; k<bytesPerPixel; k++){
            //             rightData[i][j][k] = rightImg.get(i,j,k);
            //         }
            //     }
            // }
            // Mat leftImage(leftHeight, leftWidth, CV_8U, leftData);
            // Mat midImage(midHeight, midWidth, CV_8U, midData);
            // Mat rightImage(rightHeight, rightWidth, CV_8U, rightData);

            // detector->detect(leftImage, leftKeypoints);
            // detector->detect(midImage, midKeypoints);
            // detector->detect(rightImage, rightKeypoints);


        }
};

int main(int argc, char *argv[]){

    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw" << endl;
        return 0;
    }

    ProblemTwoSolver solver;
    // solver.performStitch(argv[1], argv[2], argv[3], 483, 322, 487, 325, 489, 325, 3, argv[2]);
    solver.performStitch(argv[1], argv[2], argv[3], 322, 483, 325, 487, 325, 489, 3, argv[4]);
    
}