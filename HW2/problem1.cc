/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/
#include "common.h"

#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class ProblemOneSolver{
    public:
        float[] computeScore(Image gt, Image pred){
            int tp=0, fp=0, tn=0, fn=0;
            for (int i=0; i<gt.height; i++){
                for(int j=0; j<gt.width; j++){
                    for(int k=0; k<gt.bytesPerPixel; k++)[
                        unsigned char gt_pixel = gt.get(i,j,k);
                        unsigned char pred_pixel = pred.get(i,j,k);
                        if(gt_pixel == pred_pixel && gt_pixel==255){
                            tp++;
                        }
                        else if(gt_pixel == pred_pixel && gt_pixel==0){
                            tn++;
                        }
                        else if(gt_pixel != pred_pixel && gt_pixel == 0){
                            fp++;
                        }
                        else if(gt_pixel != pred_pixel && gt_pixel == 255){
                            fn++;
                        }
                    ]
                }
            }
            float scores[7];
            scores[0] = tp;
            scores[1] = fp;
            scores[2] = tn;
            scores[3] = fn;
            scores[4] = tp/(tp+fp);
            scores[5] = tp/(tp+fn);
            scores[6] = 2*(scores[4]*scores[5])/(scores[4]+scores[5]);
            return scores;
        }
        void performSobelEdge(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, int threshold){
            Image src(fileLocation,width, height, bytesPerPixel);
            Image gray(width, height, 1);
            Image dst(width, height, 1);
            
            for(int i=0; i<src.height; i++){
                for(int j=0; j<src.width; j++){
                    unsigned char pixel = 0.2989*(float)src.get(i,j,0) + 0.5870*(float)src.get(i,j,1) + 0.1140*(float)src.get(i,j,2);
                    gray.set(pixel, i, j, 0);
                }
            }
            FloatImage kernelX(3,3,1);
            kernelX.set(-0.25, 0, 0, 0);
            kernelX.set(0, 0, 1, 0);
            kernelX.set(0.25, 0, 2, 0);
            kernelX.set(-0.5, 1, 0, 0);
            kernelX.set(0, 1, 1, 0);
            kernelX.set(0.5, 1, 2, 0);
            kernelX.set(-0.25, 2, 0, 0);
            kernelX.set(0, 2, 1, 0);
            kernelX.set(0.25, 2, 2, 0);

            FloatImage kernelY(3,3,1);
            kernelY.set(0.25, 0, 0, 0);
            kernelY.set(0.5, 0, 1, 0);
            kernelY.set(0.25, 0, 2, 0);
            kernelY.set(0, 1, 0, 0);
            kernelY.set(0, 1, 1, 0);
            kernelY.set(0, 1, 2, 0);
            kernelY.set(-0.25, 2, 0, 0);
            kernelY.set(-0.5, 2, 1, 0);
            kernelY.set(-0.25, 2, 2, 0);
            
            for(int i=0; i<gray.height; i++){
                for(int j=0; j<gray.width; j++){
                    unsigned char pixel = sqrt(pow((float)gray.convolve(i,j,0,kernelX),2) + pow((float)gray.convolve(i,j,0,kernelY),2));
                    if (pixel>threshold)
                        dst.set(255, i, j, 0);
                    else
                        dst.set(0, i, j, 0);
                }
            }
            
            dst.saveImage(outputFileLocation);
            cout << "Sobel Edge result saved to : " << outputFileLocation << endl;

        }
        void performCannyEdge(char* fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, int thresholdHigh, int thresholdLow){
            unsigned char imageData[height][width][bytesPerPixel];
            unsigned char grayData[height][width][1];

            FILE *file;
             if(!(file=fopen(fileLocation, "rb"))){
                cout << "Cannot open file: " << fileLocation << endl;
                exit(1);
            }
            fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
            fclose(file);

            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    unsigned char pixel = 0.2989*(float)imageData[i][j][0] + 0.5870*(float)imageData[i][j][1] + 0.1140*(float)imageData[i][j][2];
                    grayData[i][j][0] = pixel;
                }
            }

            Mat src(height, width, CV_8U, grayData);
            Mat dst(height, width, CV_8U);

            Canny(src, dst, thresholdLow, thresholdHigh, 3);
            
            
        }
};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel, thresholdLow, thresholdHigh;

    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw [width] [height] [bytesPerPixel] [thresholdLow] [thresholdHigh]" << endl;
        return 0;
    }
    
    // Check if image is grayscale or color
    if(argc<5){
        width = (atoi(argv[1])==0)?580:400; //grey image
    }
    else{
        width = atoi(argv[4]);
    }

    if (argc<6){
        height = (atoi(argv[1])==0)?440:560;
    }
    else{
        height = atoi(argv[5]);
    }

    if (argc<7){
        bytesPerPixel = 1;
    }
    else{
        bytesPerPixel = atoi(argv[6]);
    }

    if (argc<8){
        thresholdLow = 127;
    }
    else{
        thresholdLow = atoi(argv[7]);
    }

    if (argc<9){
        thresholdHigh = 127;
    }
    else{
        thresholdHigh = atoi(argv[7]);
    }
    ProblemOneSolver solver;
    if (atoi(argv[1])==0){
        solver.performSobelEdge(argv[2], width, height, bytesPerPixel, argv[3], thresholdLow);
    }
    else if (atoi(argv[1])==1){
        solver.performCannyEdge(argv[2], width, height, bytesPerPixel, argv[3], thresholdLow, thresholdHigh);
    }
}