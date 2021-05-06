/*
Problem 1 - Image demosaicing and histogram manipulation
*/

#include "common.h"

class ProblemOneSolver: Image
{
    public:
        Image image;
        ProblemOneSolver(char *fileLocation, int width, int height, int bytesPerPixel){
            image.load(fileLocation, width, height, bytesPerPixel);
        }
        // void load(char *fileLocation, int width, int height, int bytesPerPixel){
        //     this->width = width;
        //     this->height = height;
        //     this->bytesPerPixel = bytesPerPixel;
        //     unsigned char imageData[height][width][bytesPerPixel];
        //     this->data = new unsigned char[width*height*bytesPerPixel];
        //     cout << "Data size: " << width*height*bytesPerPixel << endl;
        //     for (int i=0; i<width*height*bytesPerPixel; i++){
        //         this->data[i] = 0;
        //     }
        //     FILE *file;
        //     if(!(file=fopen(fileLocation, "rb"))){
        //         cout << "Cannot open file: " << fileLocation << endl;
        //         exit(1);
        //     }
        //     fread(imageData, sizeof(unsigned char), width*height*bytesPerPixel, file);
        //     fclose(file);

        //     for(int i=0; i<height; i++){
        //         for(int j=0; j<width; j++){
        //             for(int k=0; k<bytesPerPixel; k++)
        //                 this->data[i + j*height + k*height*bytesPerPixel] = imageData[i][j][k];  
        //         }
        //     }
        //     cout << "Loaded image: " << fileLocation << endl;
        // }
        int checkColor(int i, int j){
            if(i%2==0){
                if (j%2==0)
                    return GREEN;
                else
                    return BLUE;
            }
            else{
                if (j%2==0)
                    return RED;
                else
                    return GREEN;
            }
        }
        void performDemosaic(char* output_location){
            unsigned char outputData[this->height][this->width][3];
            
            for(int i=0; i<this->height; i++){
                for(int j=0;j<this->width; j++){
                    for(int k=0;k<3;k++){
                        outputData[i][j][k] = 0;
                    }
                }
            }

            for(int i=0; i<this->height; i++){
                for(int j=0; j<this->width; j++){
                    int color = checkColor(j,i);
                    

                    if (color == GREEN){
                        outputData[i][j][GREEN] = (unsigned char)get(i,j,0);
                        if (i%2==0){
                        outputData[i][j][RED] = (unsigned char)(int)(0.5 * ((float)this->get(i,j-1,0)+(float)this->get(i,j+1,0)));
                        outputData[i][j][BLUE] = (unsigned char)(int)(0.5 * ((float)this->get(i-1,j,0)+(float)this->get(i+1,j,0)));
                        }
                        else{
                        outputData[i][j][BLUE] = (unsigned char)(int)(0.5 * ((float)get(i,j-1,0)+(float)get(i,j+1,0)));
                        outputData[i][j][RED] = (unsigned char)(int)(0.5 * ((float)get(i-1,j,0)+(float)get(i+1,j,0)));

                        }
                        
                    }
                    else if (color==RED){
                        outputData[i][j][RED] = (unsigned char)get(i,j,0);
                        outputData[i][j][GREEN] = (unsigned char)(int)(0.25 * ((float)get(i,j-1,0)+(float)get(i,j+1,0)+(float)get(i-1,j,0)+(float)get(i+1,j,0)));
                        outputData[i][j][BLUE] = (unsigned char)(int)(0.25 * ((float)get(i-1,j-1,0)+(float)get(i-1,j+1,0)+(float)get(i+1,j+1,0)+(float)get(i+1,j-1,0)));
                    }
                    else if (color==BLUE){
                        outputData[i][j][BLUE] = (unsigned char)get(i,j,0);
                        outputData[i][j][GREEN] = (unsigned char)(int)(0.25 * ((float)get(i,j-1,0)+(float)get(i,j+1,0)+(float)get(i-1,j,0)+(float)get(i+1,j,0)));
                        outputData[i][j][RED] = (unsigned char)(int)(0.25 * ((float)get(i-1,j-1,0)+(float)get(i-1,j+1,0)+(float)get(i+1,j+1,0)+(float)get(i+1,j-1,0)));
                    }
                }
            }   

            FILE *file;
            if(!(file=fopen(output_location, "wb"))){
                cout << "Unable to open file: " << output_location << endl;
                exit(1);
            }
            fwrite(outputData, sizeof(unsigned char), this->width*this->height*3, file);
            fclose(file);
            cout << "Demosaicing completed!" << endl;
            this->saveImage("house_final.raw");
        }
        void histogramEqualizeA(char *file_location, char* histFileLocation="histogramMethodA.csv"){
            int counter[256];
            
            this->computeHistogram(counter);

            for(int i=1;i<256;i++){
                counter[i] += counter[i-1];
                cout << (float)counter[i]/(this->width*this->height)<<",";
            }
            for (int i=0;i<this->height;i++){
                for(int j=0;j<this->width;j++){
                    unsigned char pixel = get(i,j,0);
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
    void histogramEqualizeB(char *fileLocation, char* histFileLocation="histogramMethodB.csv"){
        int counter[256];
        
        this->computeHistogram(counter);
        
        int max = counter[0];
        for(int i=0;i<256;i++){
            if (counter[i]>max){
                max = counter[i];
            }
        }
        
        struct Point{
            int x, y;
            float val;
        };

        Point points[256][max+1];
        int storedPointsCounter[256];

        for(int i=0;i<256;i++){
            storedPointsCounter[i]=0;
            for(int j=0;j<max+1;j++){
                points[i][j].x=-1;
                points[i][j].y=-1;
                points[i][j].val=-1;
            }
        }

        for(int i=0; i<this->height; i++){
            for(int j=0; i<this->width; j++){
                unsigned char pixel = get(i,j,0);
                storedPointsCounter[(int)pixel]++;
                points[(int)pixel][storedPointsCounter[(int)pixel]].x=i;
                points[(int)pixel][storedPointsCounter[(int)pixel]].y=j;
                points[(int)pixel][storedPointsCounter[(int)pixel]].val=pixel;
            }
        }

        int perBucketPixels = (this->width*this->height)/256;
        int intensityPos=0;
        for(int i=0; i<256; i++){
            for(int j=0;j<perBucketPixels;j++){
                cout << storedPointsCounter[intensityPos] << " " << intensityPos << endl;
                if(storedPointsCounter[intensityPos]>0){
                    set(points[intensityPos][storedPointsCounter[intensityPos]].val, points[intensityPos][storedPointsCounter[intensityPos]].x, points[intensityPos][storedPointsCounter[intensityPos]].y, 0);
                    storedPointsCounter[intensityPos]--;
                }
                else{
                    intensityPos++;
                }
                if (intensityPos>255)
                    break;
            }
        }
        saveImage(fileLocation);
    }
};


int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw [width] [height] [bytesPerPixel]" << endl;
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
    

    ProblemOneSolver src(argv[2], width, height, bytesPerPixel);
    cout << width << " " << height << " " << bytesPerPixel;
    if (atoi(argv[1])==0)
        src.performDemosaic(argv[3]);
    else if (atoi(argv[1])==1)
        src.histogramEqualizeA(argv[3]);
    else if (atoi(argv[1])==2)
        src.histogramEqualizeB(argv[3]); //Change this to method B

    
    return 0;

}