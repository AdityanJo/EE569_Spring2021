/*
 Oil Painting
*/

#include "common.h"

using namespace std; 

class ProblemThreeSolver{
    public:

        void performQuantization(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, int nColors=4){
            
            Image src(fileLocation, width, height, bytesPerPixel);
            Image dst(width, height, bytesPerPixel);
            
            unsigned char topColors[bytesPerPixel][256];
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<256; i++){
                    topColors[k][i]=0;
                }
            }
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        topColors[k][(int)src.get(i,j,k)]+=1;
                    }
                }
            }
            //Sort colors by frequency
            for(int k=0; k<bytesPerPixel; k++){
                for(int i=0; i<256; i++){
                    int max = i;
                    for(int j=i+1; j<256; j++){
                        if (topColors[k][j]>topColors[k][max]){
                            topColors[k][j] = topColors[k][j] + topColors[k][max];
                            topColors[k][max] = topColors[k][j] - topColors[k][max];
                            topColors[k][j] = topColors[k][j] - topColors[k][max];
                        }
                    }
                }
            }
            
            int colorBinCounter[bytesPerPixel][nColors];
            unsigned char colorBins[bytesPerPixel][nColors];
            for(int k=0; k<bytesPerPixel; k++){
                for(int c=0; c<nColors; c++){
                    colorBins[k][c] = topColors[k][int(256/(c+1))];
                    colorBinCounter[k][c]=0;
                }
            }
            int pixelsPerBin = (width*height)/4;
            
            
            for(int k=0; k<bytesPerPixel; k++){    
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        unsigned char pixel = src.get(i,j,k);
                        float min_dist=100000;
                        int min_idx = 0;
                        for(int d_i=0; d_i<nColors; d_i++){
                            if(pow(colorBins[k][d_i]-pixel,2)<min_dist && colorBinCounter[k][d_i]<pixelsPerBin){
                                min_dist = pow(colorBins[k][d_i]-pixel, 2);
                                min_idx = d_i;
                            }
                            else if(colorBinCounter[k][d_i]>pixelsPerBin){
                                min_idx = d_i+1;
                            }
                        }
                        // cout << "Using color "<< colorBins[k][min_idx] << endl;
                        src.set((colorBins[k][min_idx]), i, j, k);
                        colorBinCounter[k][min_idx]++;
                    }
                }   
            }
            src.saveImage(outputFileLocation);
        }
};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: input_image.raw output_image.raw [width] [height] [bytesPerPixel] " << endl;
        return 0;
    }
    
    
    if(argc<4){
        width = 500; 
    }
    else{
        width = atoi(argv[3]);
    }

    if (argc<5){
        height = 400;
    }
    else{
        height = atoi(argv[4]);
    }

    if (argc<6){
        bytesPerPixel = 3;
    }
    else{
        bytesPerPixel = atoi(argv[5]);
    }
    
    ProblemThreeSolver solver;
    solver.performQuantization(argv[1], width, height, bytesPerPixel, argv[2]);    
    
    return 0;
}