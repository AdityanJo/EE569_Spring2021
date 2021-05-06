#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

#define RED 0
#define GREEN 1
#define BLUE 2

class Image{
    private:
        unsigned char *data;
        int width, height;
        int bytesPerPixel;
    public: 
    Image(char *file_location, int width, int height, int bytesPerPixel){
            this->width = width;
            this->height = height;
            this-> bytesPerPixel = bytesPerPixel;
            unsigned char imageData[width][height][bytesPerPixel];
            this->data = new unsigned char[width*height*bytesPerPixel];
            cout << "Data size: " << width*height*bytesPerPixel << endl;
            for (int i=0; i<width*height*bytesPerPixel; i++){
                this->data[i] = 0;
            }
            
            FILE *file;
            if(!(file=fopen(file_location, "rb"))){
                cout << "Cannot open file: " << file_location << endl;
                exit(1);
            }
            fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
            fclose(file);

            for(int i=0; i<width; i++){
                for(int j=0; j<height; j++){
                    for(int k=0; k<bytesPerPixel; k++)
                        this->data[i + j*width + k*width*bytesPerPixel] = imageData[i][j][k];  
                }
            }
            cout << "Loaded image: " << file_location << endl;
    }
    unsigned char get(int i, int j, int k){
        //Reflection padding cases
        if (i<0){
            i = -i;
        }
        if (j<0){
            j=-j;
        }
        if (i>=this->width){
            i = 2*(this->width-1)- i; 
        }
        if (j>=this->height){
            j = 2*(this->height-1) - j;
        }
        return this->data[i + j*this->width + k*this->width*this->bytesPerPixel];
    }
    void set(unsigned char val, int i, int j, int k){
            this->data[i + j*this->width + k*this->width*this->bytesPerPixel] = val;
    }
    void saveImage(char* file_location){
            unsigned char outputData[this->width][this->height][this->bytesPerPixel];

            for(int i=0;i<this->width;i++){
                for(int j=0;j<this->height;j++){
                    for(int k=0;k<this->bytesPerPixel;k++){
                        outputData[i][j][k] = get(i,j,k);
                    }
                }
            }

            FILE *file;
            if(!(file=fopen(file_location,"wb"))){
                cout << "Unable to load file: " << file_location;
                exit(1);
            }
            fwrite(outputData, sizeof(unsigned char), this->width*this->height*this->bytesPerPixel, file);
            fclose(file);
    }
    void methodA(char *file_location, char* histFileLocation="histogramMethodA.csv"){
        int counter[256];
        for(int i=0;i<256;i++)
            counter[i]=0;
        
        for (int i=0;i<this->width*this->height;i++)
            counter[this->data[i]]+=1;
        
        for(int i=1;i<256;i++)
            counter[i] += counter[i-1];
        for (int i=0;i<this->width;i++){
            for(int j=0;j<this->height;j++){
                unsigned char pixel = get(i,j,0);
                cout << " CDF " << (int)pixel <<" "<< counter[pixel]<< " " <<((float)counter[pixel]/((float)this->width*this->height))<<endl;
                set((unsigned char)(255*((float)counter[get(i,j,0)]/((float)this->width*this->height))),i,j,0);
            }
        }
        saveImage(file_location);

        ofstream histFile;
        histFile.open(histFileLocation);
        for(int i=0;i<256;i++){
            histFile << (int)(255*((float)counter[i]/((float)this->width*this->height))) << ",";
        }
        histFile.close();
    }
    void methodB(char *file_location){
        int counter[256];
    }
};

int main(int argc, char *argv[]){
    int bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: input_image.raw output_image.raw [bytesPerPixel]" << endl;
        return 0;
    }

    // Check if image is grayscale or color
    if(argc<4){
        bytesPerPixel = 1; //grey image
    }
    else{
        bytesPerPixel = atoi(argv[3]);
    }

    Image src(argv[1], 400, 560, bytesPerPixel);

    src.methodA(argv[2]);    
    return 0;

}