/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/

#include "common.h"

class ProblemTwoSolver{
    public:
        FloatImage createDitherMatrix(int N){
            FloatImage base(2,2,1);
            base.set(1, 0, 0, 0);
            base.set(2, 0, 1, 0);
            base.set(3, 1, 0, 0);
            base.set(0, 1, 1, 0);
            if (N==2){
                return base;
            }
            else{
                FloatImage dOne(N/2, N/2, 1);
                FloatImage dTwo(N/2, N/2, 1);
                FloatImage dThree(N/2, N/2, 1);
                FloatImage dFour(N/2, N/2, 1);

                dOne = createDitherMatrix(N/2);
                dTwo = createDitherMatrix(N/2);
                dThree = createDitherMatrix(N/2);
                dFour = createDitherMatrix(N/2);
                
                dOne.multiply(4);
                dTwo.multiply(4);
                dThree.multiply(4);
                dFour.multiply(4);

                dOne.add(1);
                dTwo.add(2);
                dThree.add(3);
                FloatImage res(dOne, dTwo, dThree, dFour);

                return res;
            }

        }
        void fixedThresholding(char* fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, int threshold){
            Image src(fileLocation, width, height, bytesPerPixel);
            Image dst(width, height, bytesPerPixel);

            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                        
                        unsigned char pixel = src.get(i,j,0);
                        cout << (int)pixel << endl;
                        if ((int)pixel<threshold and (int)pixel>=0)
                            dst.set(0, i, j, 0);
                        else
                            dst.set(255, i, j, 0);
                        
                    }
                
            }
            dst.saveImage(outputFileLocation);
        }
        void randomThresholding(char* fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation){
            Image src(fileLocation, width, height, bytesPerPixel);

            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    for(int k=0; k<bytesPerPixel; k++){
                        
                        unsigned char pixel = src.get(i,j,k);
                        int threshold = rand() % 255;
                        if (pixel<threshold and pixel>=0)
                            src.set(0, i, j, k);
                        else
                            src.set(255, i, j, k);
                    }
                }
            }
            src.saveImage(outputFileLocation);
        }
        void ditherHalfTone(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, int matSize){
            FloatImage thresholdMatrix(matSize, matSize, 1);
            
            thresholdMatrix = createDitherMatrix(matSize);
            thresholdMatrix.add(0.5);
            thresholdMatrix.multiply(255.0/(matSize*matSize));
            
            Image src(fileLocation, width, height, bytesPerPixel);
            Image dst(width, height, bytesPerPixel);

            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    unsigned char pixel = src.get(i, j, 0);
                    if (pixel>thresholdMatrix.get(i%matSize, j%matSize, 0))
                        dst.set(255, i, j, 0);
                    else
                        dst.set(0, i, j, 0);
                }
            }
            dst.saveImage(outputFileLocation);
        }
        void errorDiffusion(char* fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, float threshold, int diffuserType){
            FloatImage src(width, height, bytesPerPixel);
            src.load(fileLocation);
            src.normalize();

            if(diffuserType==0){
                for(int i=0; i<src.height; i++){
                    if (i%2==0){
                        for(int j=0; j<src.width; j++){
                            src.diffuseErrorFS(i, j, 0, threshold, false);
                        }
                    }
                    else{
                        for(int j=src.width-1; j>=0; j--){
                            src.diffuseErrorFS(i, j, 0, threshold, true);
                        }
                    }
                }
            }
            else if(diffuserType==1){

                for(int i=0; i<src.height; i++){
                    if (i%2==0){
                        for(int j=0; j<src.width; j++){
                            src.diffuseErrorJJN(i, j, 0, threshold, false);
                        }
                    }
                    else{
                        for(int j=src.width-1; j>=0; j--){
                            src.diffuseErrorJJN(i, j, 0, threshold, true);
                        }
                    }
                }
            }
            else if(diffuserType==2){
                for(int i=0; i<src.height; i++){
                    if (i%2==0){
                        for(int j=0; j<src.width; j++){
                            src.diffuseErrorStucki(i, j, 0, threshold, false);
                        }
                    }
                    else{
                        for(int j=src.width-1; j>=0; j--){
                            src.diffuseErrorStucki(i, j, 0, threshold, true);
                        }
                    }
                }
            }
            src.binarize();
            src.save8BitImage(outputFileLocation);
        }

};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel, diffuserType;
    float threshold;
    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw [width] [height] [bytesPerPixel] [threshold] [diffuserType=0/1/2]" << endl;
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
        threshold = 127;
    }
    else{
        threshold = atof(argv[7]);
    }

    if (argc<9){
        diffuserType = 0;
    }
    else{
        diffuserType = atoi(argv[8]);
    }

    ProblemTwoSolver solver;
    
    if (atoi(argv[1])==0){
        solver.fixedThresholding(argv[2], width, height, bytesPerPixel, argv[3], threshold);
    }
    else if(atoi(argv[1])==1){
        solver.randomThresholding(argv[2], width, height, bytesPerPixel, argv[3]);
    }
    else if(atoi(argv[1])==2){
        solver.ditherHalfTone(argv[2], width, height, bytesPerPixel, argv[3], threshold);
    }
    else if(atoi(argv[1])==3){
        solver.errorDiffusion(argv[2], width, height, bytesPerPixel, argv[3], threshold, diffuserType);
    }

}