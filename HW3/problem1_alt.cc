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
        float getMaxR(float theta, float max, float x_min, float x_max, float y_min, float y_max, bool use_x_axes=true){
            float beg = 0;
            float end = max;
            float mid = (beg+end)/2;
            // cout << beg-end << max << endl;
            while (abs(beg-end)>1){
                
                float x = mid*cos(theta*PI/180);
                float y = mid*sin(theta*PI/180);
                // cout << beg << " " << mid << " " << end  << " " << x_max << " " << y_max  << " " << x << "," << y << " " << theta << endl;
                
                if(x<x_min || x>x_max || y<y_min || y>y_max){
                    end=mid-1;
                }
                else{
                    beg = mid+1;
                }
                
                mid = (beg+end)/2;
                
            }
            return mid;
        }
        void performSpatialWarp(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation, char *outRecFileLocation){
            Image src(fileLocation, width, height, bytesPerPixel);
            float diag = sqrt(width*width+height*height);
            float radius = (diag+1)/2;
            
            Image warped((int)diag+1, (int)diag+1, bytesPerPixel);

            float min_x=0;
            float min_y=0;
            float max_x=0;
            float max_y=0;

            for(int i=0;i<diag+1; i++){
                for(int j=0;j<diag+1; j++){
                    // i=0;
                    // j=0;
                    float y = radius-i+0.5;
                    float x = j-radius+0.5;
                    
                    float r = sqrt(x*x+y*y);
                    float theta = atan2(y,x)*180/PI;

                    float max_r = getMaxR(theta, 233, -164, 164, -164,164);
                        
                    float warp_ratio = max_r/233;
                    float r_new = warp_ratio*r;
                    float x_target = r_new*cos(theta*PI/180);
                    float y_target = r_new*sin(theta*PI/180);
                    x_target+=164;
                    y_target+=164;
                    warped.set(src.get(329-y_target,x_target,0,false), i, j, 0);
                    warped.set(src.get(329-y_target,x_target,1,false), i, j, 1);
                    warped.set(src.get(329-y_target,x_target,2,false), i, j, 2);
                }
            }
                    
            warped.saveImage(outFileLocation);

            Image recreated(width, height, bytesPerPixel);

            for(int i=0;i<height; i++){
                for(int j=0; j<width; j++){
                    float y = width/2-i+0.5;
                    float x = j-height/2+0.5;
                    
                    float r = sqrt(x*x+y*y);
                    float theta = atan2(y,x)*180/PI;

                    float max_r = getMaxR(theta, 233, -164, 164, -164,164);
                        
                    float warp_ratio = 233/max_r;
                    float r_new = warp_ratio*r;
                    float x_target = r_new*cos(theta*PI/180);
                    float y_target = r_new*sin(theta*PI/180);
                    x_target+=233;
                    y_target+=233;
                    
                    recreated.set(warped.get(466-y_target,x_target,0,false), i, j, 0);
                    recreated.set(warped.get(466-y_target,x_target,1,false), i, j, 1);
                    recreated.set(warped.get(466-y_target,x_target,2,false), i, j, 2);
                }
            }
            recreated.saveImage(outRecFileLocation);
        }
};


int main(int argc, char *argv[]){
    int width=329, height=329, bytesPerPixel=3;

    if(argc<3){
        cout << "Usage wrong: input_image.raw output_image.raw output_recreate.raw" << endl;
        return 0;
    }

    ProblemOneSolver solver;
    // cout << width << " " << height << " " << bytesPerPixel << endl;
    solver.performSpatialWarp(argv[1], width, height, bytesPerPixel, argv[2], argv[3]);
    
}