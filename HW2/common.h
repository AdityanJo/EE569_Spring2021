/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
using namespace std;

#define RED 0
#define GREEN 1
#define BLUE 2

#define CMYW 100
#define MYGC 101
#define RGMY 102
#define KRGB 103
#define RGBM 104
#define CMGB 105

#define V_WHITE 200 
#define V_YELLOW 201
#define V_MAGENTA 202
#define V_CYAN 203
#define V_GREEN 204
#define V_RED 205
#define V_BLUE 206
#define V_BLACK 207

class FloatImage{
    public:
        float *data;
        int width, height, bytesPerPixel;
        void normalize(){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        set(get(i,j,k)/255.0, i, j, k);
                    }
                }
            }
        }
        void binarize(float threshold=0.5){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        set((get(i,j,k)>threshold)?255:0, i, j, k);
                    }
                }
            }
        }
        void convertToCMY(FloatImage dst){
            
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    float r_ = this->get(i,j,RED);
                    float g_ = this->get(i,j,GREEN);
                    float b_ = this->get(i,j,BLUE);
                    
                    r_/=255.0;
                    g_/=255.0;
                    b_/=255.0;

                    r_=1-r_;
                    g_=1-g_;
                    b_=1-b_;

                    dst.set(r_,i,j,0);
                    dst.set(g_,i,j,1);
                    dst.set(b_,i,j,2);
                }
            }
        }
        void compareImages(FloatImage src, FloatImage dst){
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        if (abs(src.get(i,j,k)-dst.get(i,j,k))>1e-5){
                            cout <<"Diff:" << src.get(i,j,k) << " " << dst.get(i,j,k) << " " << (abs(src.get(i,j,k)-dst.get(i,j,k))) << endl;
                        }
                    }
                }
            }
            
        }
        void convertToRGB(FloatImage dst){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    float r_ = this->get(i,j,RED);
                    float g_ = this->get(i,j,GREEN);
                    float b_ = this->get(i,j,BLUE);

                    r_ = 1-r_;
                    g_ = 1-g_; 
                    b_ = 1-b_;

                    r_*=255.0;
                    g_*=255.0;
                    b_*=255.0;

                    dst.set(r_,i,j,RED);
                    dst.set(g_,i,j,GREEN);
                    dst.set(b_,i,j,BLUE);
                }
            }
        }
        FloatImage(FloatImage dOne, FloatImage dTwo, FloatImage dThree, FloatImage dFour){
            // Only for dither matrix formation
            this->width = dOne.width+dTwo.width;
            this->height = dOne.height+dTwo.height;
            this->bytesPerPixel = dOne.bytesPerPixel;

            this->data = new float[width*height*bytesPerPixel];
            for (int i=0; i<width*height*bytesPerPixel; i++){
                this->data[i] = 0;
            }
            
            for(int i=0; i<dOne.height; i++){
                for(int j=0; j<dOne.width; j++){
                    set(dOne.get(i,j,0), i, j,0);
                }
            }
            for(int i=0; i<dOne.height; i++){
                for(int j=dOne.width; j<dOne.width+dTwo.width; j++){
                    set(dTwo.get(i, j-dOne.width, 0), i, j, 0);
                }
            }
            for(int i=dOne.height; i<dOne.height+dTwo.height; i++){
                for(int j=0; j<dTwo.width; j++){
                    set(dThree.get(i-dOne.height, j, 0),i, j,0);
                }
            }
            for(int i=dOne.height; i<dOne.height+dTwo.height; i++){
                for(int j=dTwo.width; j<dOne.width+dTwo.width; j++){
                    set(dFour.get(i-dOne.height, j-dTwo.height, 0), i, j, 0);
                }
            }
        }
        void load(char *fileLocation){
            unsigned char imageData[height][width][bytesPerPixel];

            FILE *file;
            if(!(file=fopen(fileLocation, "rb"))){
                cout << "Cannot open file: " << fileLocation << endl;
                exit(1);
            }
            fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
            fclose(file);
            
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        this->data[i + j*height + k*height*width] = (float)(int)imageData[i][j][k];  
                    }
                }
            }
        }
        void supressZeros(float threshold=0.00001){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        if (get(i,j,k)<=threshold)
                            set(0.0, i, j, k);
                    }
                }
            }
        }
        void multiply(float val){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        set((float)get(i,j,k)*val, i, j, k);
                    }
                }
            }
        }
        void add(float val){
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        set(get(i,j,k)+(unsigned char)val, i, j, k);
                    }
                }
            }
        }
        FloatImage(int width, int height, int bytesPerPixel){
            this->width = width;
            this->height = height;
            this->bytesPerPixel = bytesPerPixel;
            this->data = new float[width*height*bytesPerPixel];
            for (int i=0; i<width*height*bytesPerPixel; i++){
                this->data[i] = 0;
            }
        }
        const float get(int i, int j, int k){
            //Reflection padding cases
            if (i<0 or j<0 or i>height or j >width){
                return 0;
            }
            if (i<0){
                i = -i;
            }
            if (j<0){
                j=-j;
            }
            if (i>=this->height){
                i = 2*(this->height-1)- i; 
            }
            if (j>=this->width){
                j = 2*(this->width-1) - j;
            }
            
            return this->data[i + j*this->height + k*this->height*this->width];
        }
        void set(float val, int i, int j, int k){
            if (i<0 or j<0 or i>height-1 or j >width-1){
                return;
            }
               
            this->data[i + j*this->height + k*this->height*this->width] = val;
        }
        void save8BitImage(char*fileLocation){
            unsigned char outputData[this->height][this->width][this->bytesPerPixel];
            for(int k=0;k<this->bytesPerPixel;k++){
                for(int i=0;i<this->height;i++){
                    for(int j=0;j<this->width;j++){
                            outputData[i][j][k] = (unsigned char)(int)(get(i,j,k)+0.5-(get(i,j,k)<0));
                    }
                }
            }
            FILE *file;
            if(!(file=fopen(fileLocation,"wb"))){
                cout << "Unable to load file: " << fileLocation << endl;
                exit(1);
            }
            fwrite(outputData, sizeof(unsigned char), this->height*this->width*this->bytesPerPixel, file);
            fclose(file);
        }
        void saveImage(char* fileLocation){
            float outputData[this->height][this->width][this->bytesPerPixel];

            for(int i=0;i<this->height;i++){
                for(int j=0;j<this->width;j++){
                    for(int k=0;k<this->bytesPerPixel;k++){
                        outputData[i][j][k] = get(i,j,k);
                    }
                }
            }

            FILE *file;
            if(!(file=fopen(fileLocation,"wb"))){
                cout << "Unable to load file: " << fileLocation << endl;
                exit(1);
            }
            fwrite(outputData, sizeof(float), this->height*this->width*this->bytesPerPixel, file);
            fclose(file);
        }
        void diffuseErrorFSMBVQ(int i, int j, int mbvq, float threshold, bool flipped=false){
            float ch1=0.0, ch2=0.0, ch3=0.0;
            if (mbvq==V_RED){
                ch1 = 1.0;
                ch2 = 0.0;
                ch3 = 0.0;
            }
            else if (mbvq==V_GREEN){
                ch1 = 0.0;
                ch2 = 1.0;
                ch3 = 0.0;
            }
            else if (mbvq==V_BLUE){
                ch1 = 0.0;
                ch2 = 0.0;
                ch3 = 1.0;
            }
            else if (mbvq==V_MAGENTA){
                ch1 = 1.0;
                ch2 = 0.0;
                ch3 = 1.0;
            }
            else if (mbvq==V_CYAN){
                ch1 = 0;
                ch2 = 1.0;
                ch3 = 1.0;
            }
            else if (mbvq==V_YELLOW){
                ch1 = 1.0;
                ch2 = 1.0;
                ch3 = 0.0;
            }
            else if (mbvq==V_WHITE){
                ch1 = 1.0;
                ch2 = 1.0;
                ch3 = 1.0;
            }
            else if (mbvq==V_BLACK){
                ch1 = 0.0;
                ch2 = 0.0;
                ch3 = 0.0;
            }
            // float binary = (get(i,j,0)>threshold)?1.0:0.0;
            float error = get(i,j,0) - ch1;
            
            //Hardcoding cause the other way seems to produce weird outputs
            set(ch1, i, j, 0);
            if(flipped==true){
                set(get(i,j-1, 0)+0.4375*error, i,j-1,0);
                set(get(i+1,j-1, 0)+0.0625*error, i+1,j-1,0);
                set(get(i+1,j, 0)+0.3125*error, i+1,j,0);
                set(get(i+1,j+1, 0)+0.1875*error, i+1,j+1,0);
            }
            else{
                set(get(i,j+1, 0)+0.4375*error, i,j+1,0);
                set(get(i+1,j+1, 0)+0.0625*error, i+1,j+1,0);
                set(get(i+1,j, 0)+0.3125*error, i+1,j,0);
                set(get(i+1,j-1, 0)+0.1875*error, i+1,j-1,0);
            }

            error = get(i,j,1) - ch2;
            set(ch2, i, j, 1);
            
            //Hardcoding cause the other way seems to produce weird outputs
            if(flipped==true){
                set(get(i,j-1, 1)+0.4375*error, i,j-1,1);
                set(get(i+1,j-1, 1)+0.0625*error, i+1,j-1,1);
                set(get(i+1,j, 1)+0.3125*error, i+1,j,1);
                set(get(i+1,j+1, 1)+0.1875*error, i+1,j+1,1);
            }
            else{
                set(get(i,j+1, 1)+0.4375*error, i,j+1,1);
                set(get(i+1,j+1, 1)+0.0625*error, i+1,j+1,1);
                set(get(i+1,j, 1)+0.3125*error, i+1,j,1);
                set(get(i+1,j-1, 1)+0.1875*error, i+1,j-1,1);
            }
            
            error = get(i,j,2) - ch3;
            set(ch3, i, j, 2);
            
            //Hardcoding cause the other way seems to produce weird outputs
            if(flipped==true){
                set(get(i,j-1, 2)+0.4375*error, i,j-1,2);
                set(get(i+1,j-1, 2)+0.0625*error, i+1,j-1,2);
                set(get(i+1,j, 2)+0.3125*error, i+1,j,2);
                set(get(i+1,j+1, 2)+0.1875*error, i+1,j+1,2);
            }
            else{
                set(get(i,j+1, 2)+0.4375*error, i,j+1,2);
                set(get(i+1,j+1, 2)+0.0625*error, i+1,j+1,2);
                set(get(i+1,j, 2)+0.3125*error, i+1,j,2);
                set(get(i+1,j-1, 2)+0.1875*error, i+1,j-1,2);
            }
            
        }
        void diffuseErrorFS(int i, int j, int k, float threshold, bool flipped=false){
            float binary = (get(i,j,k)>threshold)?1.0:0.0;
            float error = get(i,j,k) - binary;
            //Hardcoding cause the other way seems to produce weird outputs
            set(binary, i, j, k);
            if(flipped==true){
                set(get(i,j-1, k)+0.4375*error, i,j-1,k);
                set(get(i+1,j-1, k)+0.0625*error, i+1,j-1,k);
                set(get(i+1,j, k)+0.3125*error, i+1,j,k);
                set(get(i+1,j+1, k)+0.1875*error, i+1,j+1,k);
            }
            else{
                set(get(i,j+1, k)+0.4375*error, i,j+1,k);
                set(get(i+1,j+1, k)+0.0625*error, i+1,j+1,k);
                set(get(i+1,j, k)+0.3125*error, i+1,j,k);
                set(get(i+1,j-1, k)+0.1875*error, i+1,j-1,k);
            }
        }
        void diffuseErrorJJN(int i, int j, int k, float threshold, bool flipped=false){
            float binary = (get(i,j,k)>threshold)?1.0:0.0;
            float error = get(i,j,k) - binary;

            if(flipped==true){
                set(get(i,j-1,k)+0.145833333*error, i, j-1, k);
                set(get(i,j-2,k)+0.104166667*error, i, j-2, k);
                set(get(i+1,j+2,k)+0.0625*error, i+1, j+2, k);
                set(get(i+1,j+1,k)+0.104166667*error, i+1, j+1, k);
                set(get(i+1,j,k)+0.145833333*error, i+1, j, k);
                set(get(i+1,j-1,k)+0.104166667*error, i+1, j-1, k);
                set(get(i+1,j-2,k)+0.0625*error, i+1, j-2, k);
                set(get(i+2,j+2,k)+0.020833333*error, i+2, j+2, k);
                set(get(i+2,j+1,k)+0.0625*error, i+2, j+1, k);
                set(get(i+2,j,k)+0.104166667*error, i+2, j, k);
                set(get(i+2,j-1,k)+0.0625*error, i+2, j-1, k);
                set(get(i+2,j-2,k)+0.020833333*error, i+2, j-2, k);
            }
            else{
                set(get(i,j+1,k)+0.145833333*error, i, j+1, k);
                set(get(i,j+2,k)+0.104166667*error, i, j+2, k);
                set(get(i+1,j-2,k)+0.0625*error, i+1, j-2, k);
                set(get(i+1,j-1,k)+0.104166667*error, i+1, j-1, k);
                set(get(i+1,j,k)+0.145833333*error, i+1, j, k);
                set(get(i+1,j+1,k)+0.104166667*error, i+1, j+1, k);
                set(get(i+1,j+2,k)+0.0625*error, i+1, j+2, k);
                set(get(i+2,j-2,k)+0.020833333*error, i+2, j-2, k);
                set(get(i+2,j-1,k)+0.0625*error, i+2, j-1, k);
                set(get(i+2,j,k)+0.104166667*error, i+2, j, k);
                set(get(i+2,j+1,k)+0.0625*error, i+2, j+1, k);
                set(get(i+2,j+2,k)+0.020833333*error, i+2, j+2, k);
                
            }
        }
        void diffuseErrorStucki(int i, int j, int k, float threshold, bool flipped=false){
            float binary = (get(i,j,k)>threshold)?1.0:0.0;
            float error = get(i,j,k) - binary;

            if(flipped==true){
                set(get(i,j-1,k)+0.19047619*error, i, j-1, k);
                set(get(i,j-2,k)+0.095238095*error, i, j-2, k);
                set(get(i+1,j+2,k)+0.047619048*error, i+1, j+2, k);
                set(get(i+1,j+1,k)+0.095238095*error, i+1, j+1, k);
                set(get(i+1,j,k)+0.19047619*error, i+1, j, k);
                set(get(i+1,j-1,k)+0.095238095*error, i+1, j-1, k);
                set(get(i+1,j-2,k)+0.047619048*error, i+1, j-2, k);
                set(get(i+2,j+2,k)+0.023809524*error, i+2, j+2, k);
                set(get(i+2,j+1,k)+0.047619048*error, i+2, j+1, k);
                set(get(i+2,j,k)+0.095238095*error, i+2, j, k);
                set(get(i+2,j-1,k)+0.047619048*error, i+2, j-1, k);
                set(get(i+2,j-2,k)+0.023809524*error, i+2, j-2, k);
            }
            else{
                set(get(i,j+1,k)+0.19047619*error, i, j+1, k);
                set(get(i,j+2,k)+0.095238095*error, i, j+2, k);
                set(get(i+1,j-2,k)+0.047619048*error, i+1, j-2, k);
                set(get(i+1,j-1,k)+0.095238095*error, i+1, j-1, k);
                set(get(i+1,j,k)+0.19047619*error, i+1, j, k);
                set(get(i+1,j+1,k)+0.095238095*error, i+1, j+1, k);
                set(get(i+1,j+2,k)+0.047619048*error, i+1, j+2, k);
                set(get(i+2,j-2,k)+0.023809524*error, i+2, j-2, k);
                set(get(i+2,j-1,k)+0.047619048*error, i+2, j-1, k);
                set(get(i+2,j,k)+0.095238095*error, i+2, j, k);
                set(get(i+2,j+1,k)+0.047619048*error, i+2, j+1, k);
                set(get(i+2,j+2,k)+0.023809524*error, i+2, j+2, k);
                
            }
        }
        void diffuseError(int i, int j, int k, float threshold, FloatImage matrix, bool flipped=false){
            float binary = (get(i,j,k)>threshold)?1:0;
            float error = get(i,j,k) - binary;
            for(int k_i=0; k_i<matrix.height; k_i++){
                for(int k_j=0; k_j<matrix.width; k_j++){
                    if (flipped==true){
                        if ((i-matrix.height/2+k_i)>this->height || (j-matrix.width/2+k_j)>this->width || (j-matrix.width/2+k_j)<0 || (i-matrix.height/2+k_i)<0)
                            continue;
                        else{
                            float pixel = get(i-matrix.height/2+k_i, j-matrix.width/2+k_j, k);
                            set(pixel+matrix.get(matrix.height-k_i, matrix.width-k_j, 0)*error, i-matrix.height/2+k_i, j-matrix.width/2+k_j, 0);
                        }
                    }
                    else{
                        if ((i-matrix.height/2+k_i)>this->height || (j-matrix.width/2+k_j)>this->width || (j-matrix.width/2+k_j)<0 || (i-matrix.height/2+k_i)<0)
                            continue;
                        else{
                            float pixel = get(i-matrix.height/2+k_i, j-matrix.width/2+k_j, k);
                            set(pixel+matrix.get(k_i, k_j, 0)*error, i-matrix.height/2+k_i, j-matrix.width/2+k_j, 0);
                        }
                    }
                }
            }
        }
};
class Image
{
    public:
        unsigned char *data;
        int width, height, bytesPerPixel;
        Image(int width, int height, int bytesPerPixel){
            this->width = width;
            this->height = height;
            this->bytesPerPixel = bytesPerPixel;
            unsigned char imageData[height][width][bytesPerPixel];
            this->data = new unsigned char[width*height*bytesPerPixel];
            for (int i=0; i<width*height*bytesPerPixel; i++){
                this->data[i] = 0;
            }
        }
        Image(char* fileLocation, int width, int height, int bytesPerPixel){
            this->width = width;
            this->height = height;
            this->bytesPerPixel = bytesPerPixel;
            unsigned char imageData[height][width][bytesPerPixel];
            this->data = new unsigned char[width*height*bytesPerPixel];
            cout << "Data size: " << width*height*bytesPerPixel << endl;
            for (int i=0; i<width*height*bytesPerPixel; i++){
                this->data[i] = 0;
            }
            FILE *file;
            if(!(file=fopen(fileLocation, "rb"))){
                cout << "Cannot open file: " << fileLocation << endl;
                exit(1);
            }
            fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
            fclose(file);

            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++)
                        this->data[i + j*height + k*height*width] = imageData[i][j][k];  
                }
            }
            cout << "Loaded image: " << fileLocation << endl;
        }
        const unsigned char get(int i, int j, int k, bool debug=false){
            //Reflection padding cases
            
            if (i<0){
                i = -i;
            }
            if (j<0){
                j=-j;
            }
            if (i>=this->height){
                i = 2*(this->height-1)- i; 
            }
            if (j>=this->width){
                j = 2*(this->width-1) - j;
            }
            
            return this->data[i + j*this->height + k*this->height*this->width];
        }
        void set(unsigned char val, int i, int j, int k){
            if (i<0 or j<0 or i>height-1 or j>width-1)
                return;
            this->data[i + j*this->height + k*this->height*this->width] = val;
        }
        void saveImage(char* fileLocation){
            unsigned char outputData[this->height][this->width][this->bytesPerPixel];

            for(int i=0;i<this->height;i++){
                for(int j=0;j<this->width;j++){
                    for(int k=0;k<this->bytesPerPixel;k++){
                        outputData[i][j][k] = get(i,j,k);
                    }
                }
            }

            FILE *file;
            if(!(file=fopen(fileLocation,"wb"))){
                cout << "Unable to load file: " << fileLocation << endl;
                exit(1);
            }
            fwrite(outputData, sizeof(unsigned char), this->height*this->width*this->bytesPerPixel, file);
            fclose(file);
        }
        void computeHistogram(int counter[]){
            for(int i=0;i<256;i++)
                counter[i]=0;
            
            for (int i=0;i<this->width*this->height;i++)
                counter[(int)this->data[i]]+=1;
        }
        float convolve(int i, int j, int k, FloatImage kernel){
            float sum = 0;
            for(int k_i=0; k_i<kernel.height; k_i++){
                for(int k_j=0; k_j<kernel.width; k_j++){
                    sum += ((float)get(i-kernel.height/2+k_i, j-kernel.width/2+k_j, k)) * kernel.get(k_i, k_j,0);
                }
            }
            return sum;
        }
        void filter(FloatImage kernel){
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        set(convolve(i,j,k,kernel),i,j,k);
                    }
                }
            }
        }
        void diffuseError(int i, int j, int k, int threshold, FloatImage matrix, bool flipped=false){
            int binary = (get(i,j,k)>threshold)?1:0;
            float error = get(i,j,k) - binary;
            for(int k_i=0; k_i<matrix.height; k_i++){
                for(int k_j=0; k_j<matrix.width; k_j++){
                        // cout << (int)matrix.get(k_i, k_j, 0) << " ";
                        
                    if (flipped==true){
                        if ((i-matrix.height/2+k_i)>this->height || (j-matrix.width/2+k_j)>this->width || (j-matrix.width/2+k_j)<0 || (i-matrix.height/2+k_i)<0)
                            continue;
                        else{
                            unsigned char pixel = get(i-matrix.height/2+k_i, j-matrix.width/2+k_j, k);
                            set((float)pixel/255.0+matrix.get(matrix.height-k_i, matrix.width-k_j, 0)*error, i-matrix.height/2+k_i, j-matrix.width/2+k_j, 0);
                        }
                    }
                    else{

                        if ((i-matrix.height/2+k_i)>this->height || (j-matrix.width/2+k_j)>this->width || (j-matrix.width/2+k_j)<0 || (i-matrix.height/2+k_i)<0)
                            continue;
                        else{
                            unsigned char pixel = get(i-matrix.height/2+k_i, j-matrix.width/2+k_j, k);
                            set((float)pixel/255.0+matrix.get(k_i, k_j, 0)*error, i-matrix.height/2+k_i, j-matrix.width/2+k_j, 0);
                        }
                    }

                }
                cout << endl;
            }

            // int binary = (get(i,j,k)>threshold)?255:0;
            // float error = get(i,j,k) - binary;
            // for(int k_i=0; k_i<matrix.height; k_i++){
            //     for(int k_j=0; k_j<matrix.width; k_j++){
            //         if ((i-matrix.height/2+k_i)>this->height || (j-matrix.width/2+k_j)>this->width || (j-matrix.width/2+k_j)<0 || (i-matrix.height/2+k_i)<0)
            //             continue;
            //         else{
            //             unsigned char pixel = get(i-matrix.height/2+k_i, j-matrix.width/2+k_j, k);
            //             set((float)pixel/255.0+matrix.get(k_i, k_j, 0)*error, i-matrix.height/2+k_i, j-matrix.width/2+k_j, 0);
            //         }
            //     }
            // }
        }
};
