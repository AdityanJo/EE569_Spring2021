/*
Problem 2 - Image denoising
*/

#include "common.h"
#include "assert.h"
#include "math.h"

// #include <opencv2/core.hpp>
// #include <opencv2/photo.hpp>
// #include <opencv2/highgui.hpp>

// using namespace cv;

class ProblemTwoSolver
{
    public:
        float computePSNR(Image ftr, Image ori){
            assert (ftr.width==ori.width);
            assert (ftr.height==ori.height);
            assert (ftr.bytesPerPixel==ori.bytesPerPixel);

            int width = ftr.width;
            int height = ftr.height;
            int bytesPerPixel = ftr.bytesPerPixel;
            float mse=0;
            for(int i=0;i<width*height*bytesPerPixel; i++){
                mse+=pow((ftr.data[i]-ori.data[i]),2);
            }
            mse/=width*height;
            return 10 * log10((255*255)/mse);
        }
        FloatImage createGaussianKernel(int size, float std){
            FloatImage kernel(size, size, 1);
            for(int i=0;i<size;i++){
                for(int j=0;j<size;j++){
                    float val = (1/(2*3.14*std*std))*exp(-(i*i+j*j)/2*std*std);
                    kernel.set(val,i,j,0);
                }
            }
            return kernel;
        }
        FloatImage createUniformKernel(int size){
            FloatImage kernel(size, size, 1);
            for(int i=0; i<size; i++){
                for(int j=0; j<size; j++){
                    kernel.set(1/((float)size*(float)size), i, j, 0);
                }
            }
            return kernel;
        }
        void linearDenoising(char *fileLocation, char *outputFileLocation, int width, int height, int bytesPerPixel, int filterSize=3, int filterType=0, float std=0.01){
            Image src(fileLocation, width, height, bytesPerPixel);
            Image dst(fileLocation, width, height, bytesPerPixel);
            
            int counter[256];
            src.computeHistogram(counter);
            ofstream histFile;
            histFile.open("basicDenoiseA.csv");
            for(int i=0;i<256;i++){
                histFile << counter[i] << ",";
            }
            histFile.close();

            FloatImage kernel(filterSize, filterSize, 1);
            if (filterType==0){
                kernel = createUniformKernel(filterSize);
            }
            else if (filterType==1){
                kernel = createGaussianKernel(filterSize, std);
            }
            dst.filter(kernel);
            cout << "PSNR: " << computePSNR(src, dst) << endl;
            dst.saveImage(outputFileLocation);
        }
        float computeBiFiltWeight(Image src, float i, float j, int ch,  float k, float l, float std_spatial, float std_intensity){
            return exp(-(pow(i-k,2)+pow(j-l,2))/(2*std_spatial*std_spatial) - (pow(abs((float)src.get(i,j, ch)-src.get(k, l, ch)),2))/(2*std_intensity*std_intensity));
        }
        void bilateralFiltering(char *fileLocation, char *outputFileLocation, int width, int height, int bytesPerPixel, int filterSize=3, float stdSpatial=1.0, float stdIntensity=1.0){
            Image src(fileLocation, width, height, bytesPerPixel);
            Image dst(fileLocation, width, height, bytesPerPixel);
            for(int ch=0; ch<bytesPerPixel; ch++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width;j++){
                        FloatImage kernel(filterSize, filterSize, 1);
                        float sum=0;
                        for(int k=i-filterSize/2; k<i+filterSize-1; k++){
                            for(int l=j-filterSize/2; l<j+filterSize-1; l++){
                                float w = computeBiFiltWeight(src, i, j, ch, k, l, stdSpatial, stdIntensity);
                                kernel.set((unsigned char)w,k+(filterSize/2)-i,l+(filterSize/2)-j,ch);
                                sum+=w;
                            }
                        }
                        dst.set((unsigned char)((float)dst.convolve(i,j,ch, kernel)/sum), i, j, ch);
                    }
                }
            }
            dst.saveImage(outputFileLocation);
        }

        // void nonLocalMeans(char *fileLocation, char* outputFileLocation, int width, int height, int bytesPerPixel, float h=3, int templateWindowSize=7, int searchWindowSize=21){
        //     unsigned char imageData[height][width][bytesPerPixel];
            
        //     FILE *file;
        //     if(!(file=fopen(fileLocation, "rb"))){
        //         cout << "Cannot open file: " << fileLocation << endl;
        //         exit(1);
        //     }
        //     fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
        //     fclose(file);

        //     Mat src(height, width, CV_64F, imageData);
        //     Mat dst(height, width, CV_64F);

        //     fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize);

        //     imwrite(outputFileLocation, dst);

        // }

};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw [width] [height] [bytesPerPixel] " << endl;
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
    
    ProblemTwoSolver solver;
    if (atoi(argv[1])==0){
        solver.linearDenoising(argv[2], argv[3], width, height, 1, 3, 0, 1.0);    
    }
    else if(atoi(argv[1])==1){
        solver.linearDenoising(argv[2], argv[3], width, height, 1, 3, 1, 0.5);    
    }
    else if(atoi(argv[1])==2){
        solver.bilateralFiltering(argv[2], argv[3] , width, height, 1, 3, 0.35, 0.30);    
    }
    else if(atoi(argv[1])==3){
        // solver.nonLocalMeans(argv[2], argv[3], width, height, bytesPerPixel);
    }
    
    return 0;
}