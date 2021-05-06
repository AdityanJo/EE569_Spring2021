/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/

#include "common.h"

class ProblemThreeSolver{
    public:
        void separableErrorDiffusion(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation, float threshold){
            FloatImage src(width, height, bytesPerPixel);
            FloatImage cmy(width, height, bytesPerPixel);
            FloatImage dst(width, height, bytesPerPixel);
            
            src.load(fileLocation);
            src.convertToCMY(cmy);

            for(int k=0; k<src.bytesPerPixel; k++){
                for(int i=0; i<src.height; i++){
                    if (i%2==0){
                        for(int j=0; j<src.width; j++){
                            cmy.diffuseErrorFS(i, j, k, threshold,  false);
                        }
                    }
                    else{
                        for(int j=src.width-1; j>=0; j--){
                            cmy.diffuseErrorFS(i, j, k, threshold, true);
                        }
                    }
                }
            }
            
            cmy.convertToRGB(dst);
            dst.save8BitImage(outputFileLocation);
        }   
        int getMBVQ(float r, float g, float b){
            if((r+g)>1){
                if((g+b)>1){
                    if((r+g+b)>2){
                        return CMYW;
                    }
                    else{
                        return MYGC;
                    }
                }
                else{
                    return RGMY;
                }
            }
            else{
                if(!((g+b)>1)){
                    if(!((r+g+b)>1)){
                        return KRGB;
                    }
                    else{
                        return RGBM;
                    }
                }
                else{
                    return CMGB;
                }
            }
        }
        int getNearestVertex(float r, float g, float b, int mbvq){
            int vertex=V_WHITE;
            if(mbvq==CMYW){
                vertex=V_WHITE;
                if(b<0.5){
                    if(b<=r){
                        if(b<=g){
                            vertex=V_YELLOW;
                        }
                    }
                }
                if(g<0.5){
                    if(g<=b){
                        if(g<=r){
                            vertex=V_MAGENTA;
                        }
                    }
                }
                if(r<0.5){
                    if(r<=b){
                        if(r<=g){
                            vertex=V_CYAN;
                        }
                    }
                }
            }
            else if(mbvq==MYGC){
                vertex=V_MAGENTA;
                if(g>=b){
                    if(r>=b){
                        if(r>=0.5){
                            vertex=V_YELLOW;
                        }
                        else{
                            vertex=V_GREEN;
                        }
                    }
                }
                if(g>=r){
                    if(b>=r){
                        if(b>=0.5){
                            vertex=V_CYAN;
                        }
                        else{
                            vertex=V_GREEN;
                        }
                    }
                }
            }
            else if(mbvq==RGMY){
                if(b>0.5){
                    if(r>0.5){
                        if(b>=g){
                            vertex=V_MAGENTA;
                        }
                        else{
                            vertex=V_YELLOW;
                        }
                    }
                    else{
                        if(g>b+r){
                            vertex=V_GREEN;
                        }
                        else{
                            vertex=V_MAGENTA;
                        }
                    }
                }
                else{
                    if(r>=0.5){
                        if(g>=0.5){
                            vertex = V_YELLOW;
                        }
                        else{
                            vertex = V_RED;
                        }
                    }
                    else{
                        if(r>=g){
                            vertex= V_RED;
                        }
                        else{
                            vertex=V_GREEN;
                        }
                    }
                }
            }
            else if(mbvq==KRGB){
                vertex=V_BLACK;
                if(b>0.5){
                    if(b>=r){
                        if(b>=g){
                            vertex=V_BLUE;
                        }
                    }
                }
                else if(g>0.5){
                    if(g>=b){
                        if(g>=r){
                            vertex=V_GREEN;
                        }
                    }
                }
                else if(r>0.5){
                    if(r>=b){
                        if(r>=g){
                            vertex=V_RED;
                        }
                    }
                }
            }
            else if(mbvq==RGBM){
                vertex=V_GREEN;
                if(r>g){
                    if(r>=b){
                        if(b<0.5){
                            vertex=V_RED;
                        }
                        else{
                            vertex=V_MAGENTA;
                        }
                    }
                }
                if(b>g){
                    if(b>=r){
                        if(r<0.5){
                            vertex=V_BLUE;
                        }
                        else{
                            vertex=V_MAGENTA;
                        }
                    }
                }
            }
            else if(mbvq==CMGB){
                if(b>0.5){
                    if(r>0.5){
                        if(g>=r){
                            vertex=V_CYAN;
                        }
                        else{
                            vertex=V_MAGENTA;
                        }
                    }
                    else{
                        if(g>0.5){
                            vertex=V_CYAN;
                        }
                        else{
                            vertex=V_BLUE;
                        }
                    }
                }
                else{
                    if(r>0.5){
                        if(r-g+b>=0.5){
                            vertex=V_MAGENTA;
                        }
                        else{
                            vertex=V_GREEN;
                        }
                    }
                    else{
                        if(g>=b){
                            vertex=V_GREEN;
                        }
                        else{
                            vertex=V_BLUE;
                        }
                    }
                }
            }
            return vertex;
        }
        void MBVQErrorDiffusion(char *fileLocation, int width, int height, int bytesPerPixel, char* outputFileLocation){
            FloatImage src(width, height, bytesPerPixel);
            src.load(fileLocation);
            src.multiply(1/255.0);

            for(int i=0; i<src.height; i++){
                if (i%2==0){
                    for(int j=0; j<src.width; j++){
                        int mbvq = getMBVQ(src.get(i,j,0), src.get(i,j,1), src.get(i,j,2));
                        int nearestVertex = getNearestVertex(src.get(i,j,0), src.get(i,j,1), src.get(i,j,2), mbvq);
                        src.diffuseErrorFSMBVQ(i, j, nearestVertex, 0.5,  false);
                    }
                }
                else{
                    for(int j=src.width-1; j>=0; j--){
                        int mbvq = getMBVQ(src.get(i,j,0), src.get(i,j,1), src.get(i,j,2));
                        int nearestVertex = getNearestVertex(src.get(i,j,0), src.get(i,j,1), src.get(i,j,2), mbvq);
                        src.diffuseErrorFSMBVQ(i, j, nearestVertex, 0.5, true);
                    }
                }
            }
            src.multiply(255.0);
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
        threshold = 0.5;
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

    ProblemThreeSolver solver;
    
    if(atoi(argv[1])==0){
        solver.separableErrorDiffusion(argv[2], width, height, bytesPerPixel, argv[3], threshold);
    }
    else if(atoi(argv[1])==1){
        cout << " MBVQ";
        solver.MBVQErrorDiffusion(argv[2], width, height, bytesPerPixel, argv[3]);
    }
}
    