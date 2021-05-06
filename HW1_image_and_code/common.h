#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define RED 0
#define GREEN 1
#define BLUE 2

class FloatImage{
    public:
        float *data;
        int width, height, bytesPerPixel;
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
            
            return this->data[i + j*this->height + k*this->height*this->bytesPerPixel];
        }
        void set(float val, int i, int j, int k){
            this->data[i + j*this->height + k*this->height*this->bytesPerPixel] = val;
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
                        this->data[i + j*height + k*height*bytesPerPixel] = imageData[i][j][k];  
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
            
            return this->data[i + j*this->height + k*this->height*this->bytesPerPixel];
        }
        void set(unsigned char val, int i, int j, int k){
            this->data[i + j*this->height + k*this->height*this->bytesPerPixel] = val;
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
        unsigned char convolve(int i, int j, int k, FloatImage kernel){
            float sum = 0;
            for(int k_i=0; k_i<kernel.height; k_i++){
                for(int k_j=0; k_j<kernel.width; k_j++){
                    sum += get(i-kernel.height/2+k_i, j-kernel.width/2+k_j, k) * kernel.get(k_i, k_j,0);
                }
            }
            return (unsigned char)sum;
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
};