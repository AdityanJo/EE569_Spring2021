/*
Problem 1 - Image demosaicing and histogram equalization
*/

#include "common.h"
#include <vector>

using namespace std; 
class ProblemOneSolver{
    public:
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

    void performDemosaic(char *fileLocation, int width, int height, int bytesPerPixel, char* outputLocation){
        Image src(fileLocation, width, height, bytesPerPixel);
        
        src.saveImage("house_final.raw");
        unsigned char outputData[height][width][3];

        for(int i=0; i<height; i++){
            for(int j=0;j<width; j++){
                for(int k=0;k<3;k++){
                    outputData[i][j][k] = 0;
                }
            }
        }

        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                int color = checkColor(j,i);
                
                if (color == GREEN){
                    outputData[i][j][GREEN] = (unsigned char)src.get(i,j,0);
                    if (i%2==0){
                    outputData[i][j][RED] = (unsigned char)(0.5 * ((float)src.get(i,j-1,0)+(float)src.get(i,j+1,0)));
                    outputData[i][j][BLUE] = (unsigned char)(0.5 * ((float)src.get(i-1,j,0)+(float)src.get(i+1,j,0)));
                    }
                    else{
                    outputData[i][j][BLUE] = (unsigned char)(0.5 * ((float)src.get(i,j-1,0)+(float)src.get(i,j+1,0)));
                    outputData[i][j][RED] = (unsigned char)(0.5 * ((float)src.get(i-1,j,0)+(float)src.get(i+1,j,0)));
                    }
                    
                }
                else if (color==RED){
                    outputData[i][j][RED] = (unsigned char)src.get(i,j,0);
                    outputData[i][j][GREEN] = (unsigned char)(0.25 * ((float)src.get(i,j-1,0)+(float)src.get(i,j+1,0)+(float)src.get(i-1,j,0)+(float)src.get(i+1,j,0)));
                    outputData[i][j][BLUE] = (unsigned char)(0.25 * ((float)src.get(i-1,j-1,0)+(float)src.get(i-1,j+1,0)+(float)src.get(i+1,j+1,0)+(float)src.get(i+1,j-1,0)));
                }
                else if (color==BLUE){
                    outputData[i][j][BLUE] = (unsigned char)src.get(i,j,0);
                    outputData[i][j][GREEN] = (unsigned char)(0.25 * ((float)src.get(i,j-1,0)+(float)src.get(i,j+1,0)+(float)src.get(i-1,j,0)+(float)src.get(i+1,j,0)));
                    outputData[i][j][RED] = (unsigned char)(0.25 * ((float)src.get(i-1,j-1,0)+(float)src.get(i-1,j+1,0)+(float)src.get(i+1,j+1,0)+(float)src.get(i+1,j-1,0)));
                }
            }
        }

        FILE *file;
        if(!(file=fopen(outputLocation, "wb"))){
            cout << "Unable to open file: " << outputLocation << endl;
            exit(1); 
        }
        fwrite(outputData, sizeof(unsigned char), width*height*3, file);
        fclose(file);
        cout << "Written result to " << outputLocation << endl;
        cout << "Demosaicing completed!" << endl;
        
    }

    void histogramEqualizeA(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, char* histFileLocation="histogramMethodA.csv"){
            int counter[256];
            
            Image src(fileLocation, width, height, bytesPerPixel);

            src.computeHistogram(counter);

            for(int i=1;i<256;i++){
                counter[i] += counter[i-1];
            }
            for (int i=0;i<height;i++){
                for(int j=0;j<width;j++){
                    unsigned char pixel = src.get(i,j,0);
                    src.set((unsigned char)(255*((float)counter[src.get(i,j,0)]/((float)width*height))),i,j,0);
                }
            }
            src.saveImage(outputFileLocation);

            ofstream histFile;
            histFile.open(histFileLocation);
            for(int i=0;i<256;i++){
                histFile << (int)(255*((float)counter[i]/((float)width*height))) << ",";
            }
            histFile.close();
            cout << "Histogram method A complete" << endl;
    }

    void histogramEqualizeB(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, char* histFileLocation="histogramMethodB.csv"){
        Image src(fileLocation, width, height, bytesPerPixel);
        
        int counter[256];

        src.computeHistogram(counter);
        
        int max = counter[0];
        for(int i=0;i<256;i++){
            if (counter[i]>max){
                max = counter[i];
            }
        }
        
        typedef std::vector<int> RowVec;
        typedef std::vector<RowVec> Matrix;

        Matrix pointsX(256, RowVec(max+1));
        Matrix pointsY(256, RowVec(max+1));
        Matrix vals(256, RowVec(max+1));
        
        int storedPointsCounter[256];

        for(int i=0;i<256;i++){
            storedPointsCounter[i]=0;
            for(int j=0;j<max+1;j++){
                pointsX[i][j]=-1;
                pointsY[i][j]=-1;
                vals[i][j]=-1;
            }
        }


        for(int i=0; i<src.height; i++){
            for(int j=0; j<src.width; j++){
                unsigned char pixel = src.get(i,j,0);
                storedPointsCounter[(int)pixel]++;
                pointsX[(int)pixel][storedPointsCounter[(int)pixel]]=i;
                pointsY[(int)pixel][storedPointsCounter[(int)pixel]]=j;
                vals[(int)pixel][storedPointsCounter[(int)pixel]]=pixel;
            }
        }

        int perBucketPixels = (width*height)/256;
        int intensityPos=0;
        for(int i=0; i<256; i++){
            for(int j=0;j<perBucketPixels;j++){
                if(storedPointsCounter[intensityPos]>0){
                    src.set(i, pointsX[intensityPos][storedPointsCounter[intensityPos]], pointsY[intensityPos][storedPointsCounter[intensityPos]], 0);
                    storedPointsCounter[intensityPos]--;
                }
                else{
                    intensityPos++;
                }
                if (intensityPos>255)
                    break;
            }
        }
        ofstream histFile;
        histFile.open(histFileLocation);
        src.computeHistogram(counter);

        for(int i=1;i<256;i++){
            counter[i] += counter[i-1];
            histFile << ((float)(counter[i])/(width*height)) << ",";
            
        }
        histFile.close();

        src.saveImage(outputFileLocation);
        cout << "Histogram method B complete" << endl;
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
    

    ProblemOneSolver solver;

    if (atoi(argv[1])==0)
        solver.performDemosaic(argv[2], width, height, bytesPerPixel, argv[3]);
    else if (atoi(argv[1])==1)
        solver.histogramEqualizeA(argv[2], width, height, bytesPerPixel, argv[3]);
    else if (atoi(argv[1])==2)
        solver.histogramEqualizeB(argv[2], width, height, bytesPerPixel, argv[3]); //Change this to method B

    return 0;

}