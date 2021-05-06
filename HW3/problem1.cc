/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/

#include "common.h"
#include "math.h"

#define PI 3.14
class ProblemOneSolver{
    public:
        void convertImgToCartesian(int j, int k, int width, int height, float res[2]){
            res[0] = k-0.5;
            res[1] = height+0.5-j;
        }
        void convertCartesianToImg(float xk, float yj, int width, int height, float res[2]){
            res[0] = xk+0.5;
            res[1] = height+0.5-yj;
        }
        void convertImgMatrixToCartesian(float matrix[3][3], int width, int height){
            float res[2] = {0,0};

            convertImgToCartesian(matrix[0][0], matrix[1][0], width, height, res);
            matrix[0][0] = res[0];
            matrix[1][0] = res[1];

            convertImgToCartesian(matrix[0][1], matrix[1][1], width, height, res);
            matrix[0][1] = res[0];
            matrix[1][1] = res[1];

            convertImgToCartesian(matrix[0][2], matrix[1][2], width, height, res);
            matrix[0][2] = res[0];
            matrix[1][2] = res[1];
        }
        void convertCartesianMatrixToImg(float matrix[3][3], int width, int height){
            float res[2] = {0,0};

            convertCartesianToImg(matrix[0][0], matrix[1][0], width, height, res);
            matrix[0][0] = res[0];
            matrix[1][0] = res[1];

            convertCartesianToImg(matrix[0][1], matrix[1][1], width, height, res);
            matrix[0][1] = res[0];
            matrix[1][1] = res[1];

            convertCartesianToImg(matrix[0][2], matrix[1][2], width, height, res);
            matrix[0][2] = res[0];
            matrix[1][2] = res[1];
        }
        void invertMatrix(float matrix[3][3], float matrixInv[3][3]){
            float determinant = matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1]) - matrix[0][1] * (matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0]) + matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]);
            // cout << "Determinant : " << determinant << endl;
            if (determinant==0){
                cout << "Non-invertible matrix with determinant:" << determinant << endl;
                return;
            }
            float res[3][3];

            res[0][0] = matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1];
            res[0][1] = -1*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0]);
            res[0][2] = matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0];
            res[1][0] = -1*(matrix[0][1]*matrix[2][2]-matrix[0][2]*matrix[2][1]);
            res[1][1] = matrix[0][0]*matrix[2][2]-matrix[0][2]*matrix[2][0];
            res[1][2] = -1*(matrix[0][0]*matrix[2][1]-matrix[0][1]*matrix[2][0]);
            res[2][0] = matrix[0][1]*matrix[1][2]-matrix[0][2]*matrix[1][1];
            res[2][1] = -1*(matrix[0][0]*matrix[1][2]-matrix[0][2]*matrix[1][0]);
            res[2][2] = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];

            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    matrixInv[j][i] = (float)res[i][j]/determinant;
                }
            }

        }
        void multiplyMatrices(float a[3][3], float b[3][3], float c[3][3]){
            for(int i=0; i<3;i++){
                for(int j=0; j<3;j++){
                    c[i][j]=0.0;
                }
            }
            for(int i=0;i<3;i++){
                for(int j=0; j<3; j++){
                    for(int k=0; k<3;k++){
                        c[i][j] += a[i][k]*b[k][j];
                        // cout << c[i][j] << " " << a[i][k] << " " << b[k][j] << endl;
                    }
                    // cout << "==" << c[i][j] << endl;
                }
            }
        }
        void computeForwardTransform(float x_in[3][3], float x_out[3][3], float T[3][3]){
            float x_in_inv[3][3];
            invertMatrix(x_in, x_in_inv);

            // cout << x_in_inv[0][0] << " " << x_in_inv[0][1] << " " << x_in_inv[0][2] << " " << \
            //         x_in_inv[1][0] << " " << x_in_inv[1][1] << " " << x_in_inv[1][2] << " " << \
            //         x_in_inv[2][0] << " " << x_in_inv[2][1] << " " << x_in_inv[2][2] << endl;

            // cout << x_out[0][0] << " " << x_out[0][1] << " " << x_out[0][2] << " " << \
            //         x_out[1][0] << " " << x_out[1][1] << " " << x_out[1][2] << " " << \
            //         x_out[2][0] << " " << x_out[2][1] << " " << x_out[2][2] << endl;

            multiplyMatrices(x_out, x_in_inv, T);

        }
        void generateForwardTransformTriangles(int srcWidth, int srcHeight, int dstWidth, int dstHeight, \
                float tOne[3][3], float tTwo[3][3], float tThree[3][3], float tFour[3][3], float tFive[3][3], \
                float tSix[3][3], float tSeven[3][3], float tEight[3][3]){
            float srcOne[3][3] = {
                    {0,0,165},
                    {0,165,165},
                    {1,1,1}
                };

                // {68.25,0,233},
                // {68.25,233,233},
                // {1,1,1}

            float srcTwo[3][3] = {
                    {0,0,165},
                    {165,329,165},
                    {1,1,1}
                };
            float srcThree[3][3] = {
                    {0,165,165},
                    {329,329,165},
                    {1,1,1}
                };
            float srcFour[3][3] = {
                    {165,329,165},
                    {329,329,165},
                    {1,1,1}
                };
            float srcFive[3][3] = {
                    {329,329,165},
                    {329,165,165},
                    {1,1,1}
                };
            float srcSix[3][3] = {
                    {329,329,165},
                    {165,0,165},
                    {1,1,1}
                };
            float srcSeven[3][3] = {
                    {329,165,165},
                    {0,0,165},
                    {1,1,1}
                };
            float srcEight[3][3] = {
                    {165,0,165},
                    {0,0,165},
                    {1,1,1}
                };
            convertImgMatrixToCartesian(srcOne, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcTwo, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcThree, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcFour, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcFive, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcSix, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcSeven, srcWidth, srcHeight);
            convertImgMatrixToCartesian(srcEight, srcWidth, srcHeight);


            float dstOne[3][3] = {
                    {68.25,0,233},
                    {68.25,233,233},
                    {1,1,1}
                };
            float dstTwo[3][3] = {
                    {0,397.756,233},
                    {233,68.25,233},
                    {1,1,1}
                };
            float dstThree[3][3] = {
                    {397.756,233,233},
                    {68.25,466,233},
                    {1,1,1}
                };
            float dstFour[3][3] = {
                    {233,397.756,233},
                    {466,397.756,233},
                    {1,1,1}
                };
            float dstFive[3][3] = {
                    {397.756,466,233},
                    {397.756,233,233},
                    {1,1,1}
                };
            float dstSix[3][3] = {
                    {466,68.25,233},
                    {233,397.756,233},
                    {1,1,1}
                };
            float dstSeven[3][3] = {
                    {68.25,233,233},
                    {397.756,0,233},
                    {1,1,1}
                };
            float dstEight[3][3] = {
                    {233,68.25,233},
                    {0,68.25,23},
                    {1,1,1}
                };


            convertImgMatrixToCartesian(dstOne, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstTwo, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstThree, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstFour, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstFive, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstSix, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstSeven, dstWidth, dstHeight);
            convertImgMatrixToCartesian(dstEight, dstWidth, dstHeight);


            cout << srcOne[0][0] << " " << srcOne[0][1] << " " << srcOne[0][2] << " " << \
                    srcOne[1][0] << " " << srcOne[1][1] << " " << srcOne[1][2] << " " << \
                    srcOne[2][0] << " " << srcOne[2][1] << " " << srcOne[2][2] << endl;

            cout << dstOne[0][0] << " " << dstOne[0][1] << " " << dstOne[0][2] << " " << \
                    dstOne[1][0] << " " << dstOne[1][1] << " " << dstOne[1][2] << " " << \
                    dstOne[2][0] << " " << dstOne[2][1] << " " << dstOne[2][2] << endl;

            computeForwardTransform(srcOne, dstOne, tOne);
            computeForwardTransform(srcTwo, dstTwo, tTwo);
            computeForwardTransform(srcThree, dstThree, tThree);
            computeForwardTransform(srcFour, dstFour, tFour);
            computeForwardTransform(srcFive, dstFive, tFive);
            computeForwardTransform(srcSix, dstSix, tSix);
            computeForwardTransform(srcSeven, dstSeven, tSeven);
            computeForwardTransform(srcEight, dstEight, tEight);



        }
        void forwardTransformCoords(float triTransform[3][3], float coords[2], float res[2]){
            res[0] = triTransform[0][0]*coords[0] + triTransform[0][1]*coords[1] + 1*triTransform[0][2];
            res[1] = triTransform[1][0]*coords[0] + triTransform[1][1]*coords[1] + 1*triTransform[1][2];
            // cout << coords[0] << " " << coords[1] << " " << res[0] << " " << res[1] << endl;

        }
        void performSpatialWarp(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
            Image src(fileLocation, width, height, bytesPerPixel);
            float diag = sqrt(width*width+height*height);
            float radius = (diag+1)/2;
            cout << diag << endl;
            Image warped((int)diag+1, (int)diag+1, bytesPerPixel);
            
            int origin = radius;
            
            float triOne[3][3] = {{0.4136, 0, 164.4628},{0.4136,1.4121,-66.8271},{0,0,1}};
            float triTwo[3][3] = {{-0.4161, 0, 300.952},{-2.425,1.4121,400.174},{0,0,1}};
            float triThree[3][3] = {{1.421, -1.026, 299.1201},{0,-0.9985,397.7567},{0,0,1}};
            float triFour[3][3] = {{1.4287, 0.4131, -0.6916},{0, 1.0046, 68.2416},{0,0,1}};
            float triFive[3][3] = {{1.004, 0, 0.6724},{0.4161, 1.4207, -68.66},{0,0,1}};
            float triSix[3][3] = {{-1.004, 0, 397.758},{-2.4253, 1.4207, 398.762},{0,0,1}};
            float triSeven[3][3] = {{1.4121, -2.4253, 399.1747},{0,-1.004,398.76},{0,0,1}};
            float triEight[3][3] = {{-1.4121, 0.9985, -164.05},{0,0.9985,69.239},{0,0,1}};
            
            generateForwardTransformTriangles(width, height, diag+1, diag+1, triOne, triTwo, triThree, triFour, triFive, triSix, triSeven, triEight);
            
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){

                    float theta = atan2(((float)radius-i),(float)(radius-j))*180/PI;
                    
                    if(theta>0 && theta<=45){
                        //triThree
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triThree, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);

                    }
                    else if(theta>45 && theta<=90){
                        //triTwo
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triTwo, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>90 && theta<=135){
                        //triOne
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triOne, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>135 && theta<=180){
                        //triEight
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triEight, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>-45 && theta<=0){
                        //triFour
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triFour, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>-90 && theta<=-45){
                        //triFive
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triFive, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>-135 && theta<=-90){
                        //triSix
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triSix, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                    else if(theta>-180 && theta<=-135){
                        //triSeven
                        float coords[2],warpedCoords[2], warpedImgCoords[2];
                        convertImgToCartesian(i,j,width, height, coords);
                        forwardTransformCoords(triSeven, coords, warpedCoords);
                        convertCartesianToImg(warpedCoords[0], warpedCoords[1], diag+1, diag+1, warpedImgCoords);
                        
                        warped.set(src.get(i,j,0),warpedImgCoords[1], warpedImgCoords[0], 0);
                        warped.set(src.get(i,j,1),warpedImgCoords[1], warpedImgCoords[0], 1);
                        warped.set(src.get(i,j,2),warpedImgCoords[1], warpedImgCoords[0], 2);
                    }
                   
                }
            }
            warped.saveImage(outFileLocation);
            
        }
};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: input_image.raw output_image.raw [width] [height] [bytesPerPixel]" << endl;
        return 0;
    }
    
    // Check if image is grayscale or color
    if(argc<4){
        width = (atoi(argv[1])==0)?580:400; //grey image
    }
    else{
        width = atoi(argv[3]);
    }

    if (argc<5){
        height = (atoi(argv[1])==0)?440:560;
    }
    else{
        height = atoi(argv[4]);
    }

    if (argc<6){
        bytesPerPixel = 1;
    }
    else{
        bytesPerPixel = atoi(argv[5]);
    }

    ProblemOneSolver solver;
    // cout << width << " " << height << " " << bytesPerPixel << endl;
    solver.performSpatialWarp(argv[1], width, height, bytesPerPixel, argv[2]);
    
}