/*
Adityan Jothi
USC ID 8162222801
jothi@usc.edu
*/

#include "common.h"
#include "string.h"
#include <stdio.h>

class ProblemThreeSolver{
    public:
    bool compareImages(BoolImage im_a, BoolImage im_b){
        bool diffFlag=true;
        for(int i=0;i<im_a.height;i++){
            for(int j=0; j<im_b.width;j++){
                if(im_a.get(i,j,0)!=im_b.get(i,j,0)){
                        diffFlag=false;
                }
            }
        }
        return diffFlag;
    }
    // bool matchPattern(bool pattern[8], bool patternRef[8]){
    //     bool match=true;
    //     for(int i=0;i<8;i++){
    //         // cout << pattern[i] ;
    //         match = match && (pattern[i]==patternRef[i]);
    //     }
    //     // cout << " ";
    //     // for(int i=0;i<8;i++){
    //         // cout << patternRef[i];
    //     // }
    //     // cout << " " << match << endl;
    //     return match;
    // }
    bool matchPatternStrFull(bool pattern[9], string patternRef){
        bool match=true;
        for(int i=0;i<9;i++){
            // cout << pattern[i] ;
            if (patternRef[i]=='1' && pattern[i]==true){
                match = match && true;
            }
            else if(patternRef[i]=='0' && pattern[i]==false){
                match = match && true;
            }
            else{
                match = match && false;
            }
        }
        return match;
    }
    // bool matchPatternStr(bool pattern[8], string patternRef){
    //     bool match=true;
    //     for(int i=0;i<8;i++){
    //         // cout << pattern[i] ;
    //         if (patternRef[i]=='1' && pattern[i]==true){
    //             match = match && true;
    //         }
    //         else if(patternRef[i]=='0' && pattern[i]==false){
    //             match = match && true;
    //         }
    //         else{
    //             match = match && false;
    //         }
    //     }
    //     // cout << " ";
    //     // for(int i=0;i<8;i++){
    //         // cout << patternRef[i];
    //     // }
    //     // cout << " " << match << endl;
    //     return match;
    // }
    // bool matchPatternStageTwo(bool src[8], string lookup){
    //     bool a_u_b_u_c = false;
    //     bool match = true;
    //     bool abcpresent_ = false;
    //     // cout << endl;
    //     for(int i=0; i<8; i++){
    //         // cout << src[i] ;
    //         if(lookup[i]=='M' && src[i]!=true){
    //             match= match && false;
    //         }
    //         else{
    //             match= match && true;
    //         }
    //         if(lookup[i]=='A' || lookup[i]=='B' || lookup[i]=='C'){
    //             a_u_b_u_c = a_u_b_u_c || src[i];
    //             abcpresent_ = true;
    //         }
    //     }
    //     // cout << " " << lookup <<  " " << match << " " << a_u_b_u_c << endl; 
    //     if(abcpresent_){
    //         return match && a_u_b_u_c;
    //     }
    //     else{
    //         return match;
    //     }
    // }
    bool matchPatternStageTwoFull(bool src[9], string lookup){
        bool a_u_b_u_c = false;
        bool match = true;
        bool abcpresent_ = false;
        // cout << endl;
        for(int i=0; i<9; i++){
            // cout << src[i] ;
            if(lookup[i]=='M' && src[i]!=true){
                match= match && false;
            }
            else{
                match= match && true;
            }
            if(lookup[i]=='A' || lookup[i]=='B' || lookup[i]=='C'){
                a_u_b_u_c = a_u_b_u_c || src[i];
                abcpresent_ = true;
            }
            if(lookup[i]=='0'&&src[i]!=false){
                match = match && false;
            }
            else{
                match = match && true;
            }
        }
        // cout << " " << lookup <<  " " << match << " " << a_u_b_u_c << endl; 
        if(abcpresent_){
            return match && a_u_b_u_c;
        }
        else{
            return match;
        }
    }
    void applyMorphOp(BoolImage src, char *outFileLocation, int opType, bool count_dots){
        cout << opType << endl;
        BoolImage M(src.width, src.height, src.bytesPerPixel);
        BoolImage G(src.width, src.height, src.bytesPerPixel);
        BoolImage res(src.width, src.height, src.bytesPerPixel);
        BoolImage prev(src.width, src.height, src.bytesPerPixel);
        prev.copyFrom(src);
        int iteration_counter = 0;

        string shrinkPatternsStrFull[58]={
            "001010000",
            "100010000",
            "000010100",
            "000010001",
            "000011000",
            "010010000",
            "000110000",
            "000010010",
            "001011000",
            "011010000",
            "110010000",
            "100110000",
            "000110100",
            "000010110",
            "000010011",
            "000011001",
            "001011001",
            "111010000",
            "100110100",
            "000010111",
            "110011000",
            "010011001",
            "011110000",
            "001011010",
            "011011000",
            "110110000",
            "000110110",
            "000011011",
            "110011001",
            "011110100",
            "111011000",
            "011011001",
            "111110000",
            "110110100",
            "100110110",
            "000011111",
            "001011011",
            "111011001",
            "111110100",
            "100110111",
            "001011111",
            "011011011",
            "111111000",
            "110110110",
            "000111111",
            "111011011",
            "011011111",
            "111111100",
            "111111001",
            "111110110",
            "110110111",
            "100111111",
            "001111111",
            "111011111",
            "111111101",
            "111110111",
            "101111111"
            };

        string thinFull[46]={
            "010011000",
            "010110000",
            "000110010",
            "000011010",
            "001011001",
            "111010000",
            "100110100",
            "000010111",
            "110011000",
            "010011001",
            "011110000",
            "001011010",
            "011011000",
            "110110000",
            "000110110",
            "000011011",
            "110011001",
            "011110100",
            "111011000",
            "011011001",
            "111110000",
            "110110100",
            "100110110",
            "000110111",
            "000011111",
            "001011011",
            "111011001",
            "111110100",
            "100110111",
            "001011111",
            "011011011",
            "111111000",
            "110110110",
            "000111111",
            "111011011",
            "011011111",
            "111111100",
            "111111001",
            "111110110",
            "110110111",
            "100111111",
            "001111111",
            "111011111",
            "111111101",
            "111110111",
            "101111111"
        };

        string skeletonizeFull[40]={
            "010011000",
            "010110000",
            "000110010",
            "000011010",
            "001011001",
            "111010000",
            "100110100",
            "000010111",
            "111011000",
            "011011001",
            "111110000",
            "110110100",
            "100110110",
            "000110111",
            "000011111",
            "001011011",
            "111011001",
            "111110100",
            "100110111",
            "001011111",
            "011011011",
            "111111000",
            "110110110",
            "000111111",
            "111011011",
            "011011111",
            "111111100",
            "111111001",
            "111110110",
            "110110111",
            "100111111",
            "001111111",
            "111011111",
            "111111101",
            "111110111",
            "101111111",
            "111111011",
            "111111110",
            "110111111",
            "011111111"
        };


        string shrinkThinUnconditionFull[38]={
            "DDDDMMDMM",
            "00M0M0000",
            "M000M0000",

            "0000M00M0",
            "0000MM000",
            
            "00M0MM000",
            "0MM0M0000",
            "MM00M0000",
            "M00MM0000",
            
            "000MM0M00",
            "0000M0MM0",
            "0000M00MM",
            "0000MM00M",

            "0MMMM0000",
            "MM00MM000",
            "0M00MM00M",
            "00M0MM0M0",

            "0AM0MBM00",
            "MB0AM000M",
            "00MAM0MB0",
            "M000MB0AM",
            
            "MMDMMDDDD",

            "DM0MMMD00",
            "0MDMMM00D",
            "00DMMM0MD",
            "D00MMMDM0",

            "DMDMM00M0",
            "0M0MM0DMD",
            "0M00MMDMD",
            "DMD0MM0M0",

            "MDMDMDABC",
            "MDCDMBMDA",
            "CBADMDMDM",
            "ADMBMDCDM",

            "DM00MMM0D",
            "0MDMM0D0M",
            "D0MMM00MD",
            "M0D0MMDM0"
        };
        string skeletonizeUnconditional[26]={
            // "0000M000M",
            // "0000M0M00",
            // "00M0M0000",
            // "M000M0000",
            // "0000M00M0",
            // "0000MM000",
            // "000MM0000",
            // "0M00M0000",
            // "0M00MM000",
            // "0M0MM0000",
            // "0000MM0M0",
            // "000MM00M0",
            // "MMDMMDDDD",
            // "DDDDMMDMM",
            // "DMDMMMDDD",
            // "DMDMMDDMD",
            // "DDDMMMDMD",
            // "DMDDMMDMD",
            // "MDMDMDABC",
            // "MDCDMBMDA",
            // "CBADMDMDM",
            // "ADMBMDCDM",
            // "DM00MMM0D",
            // "0MDMM0D0M",
            // "D0MMM00MD",
            // "M0D0MMDM0",

            "0000M000M",
            "0000M0M00",
            "00M0M0000",
            "M000M0000",
            "0000M00M0",
            "0000MM000",
            "000MM0000",
            "0M00M0000",
            "0M00MM000",
            "0M0MM0000",
            "0000MM0M0",
            "000MM00M0",
            "MMDMMDDDD",
            "DDDDMMDMM",
            "DMDMMMDDD",
            "DMDMMDDMD",
            "DDDMMMDMD",
            "DMDDMMDMD",
            "MDMDMDABC",
            "MDCDMBMDA",
            "CBADMDMDM",
            "ADMBMDCDM",
            "DM00MMM0D",
            "0MDMM0D0M",
            "D0MMM00MD",
            "M0D0MMDM0"
        };
        // int dots=0;
        // if(count_dots)
        //     dots = prev.computeDots();

        do{
            
            if(iteration_counter!=0){
                prev.copyFrom(G);
            }
            for(int i=0;i<src.height; i++){
                for(int j=0;j<src.width; j++){
                    
                    bool pattern[9];
                    prev.getBitPatternMorphOpBoolFull(i,j,pattern);

                    if(opType==1){
                        for(int t_i=0; t_i<58; t_i++){
                            // if(prev.get(i,j,0)==false){
                            //     M.set(false,i,j,0);
                            //     break;
                            // }
                                // M.set(false, i,j,0);
                            // cout << endl << "Pattern " << t_i << endl;
                            if(matchPatternStrFull(pattern, shrinkPatternsStrFull[t_i])&&prev.get(i,j,0)){
                                M.set(true,i,j,0);
                                break;
                            }
                            else{
                                M.set(false, i,j,0);
                            }
                        }
                    }
                    else if(opType==2){
                        for(int t_i=0; t_i<46; t_i++){
                            // if(prev.get(i,j,0)==false){
                            //     M.set(false,i,j,0);
                            //     break;
                            // }
                                // M.set(false, i,j,0);
                            // cout << endl << "Pattern " << t_i << endl;
                            if(matchPatternStrFull(pattern, thinFull[t_i])&&prev.get(i,j,0)){
                                M.set(true,i,j,0);
                                break;
                            }
                            else{
                                M.set(false,i,j,0);
                            }
                        }
                    }
                    else if(opType==3){
                        for(int t_i=0; t_i<46; t_i++){
                            // if(prev.get(i,j,0)==false){
                            //     M.set(false,i,j,0);
                            //     break;
                            // }
                                // M.set(false, i,j,0);
                            // cout << endl << "Pattern " << t_i << endl;
                            if(matchPatternStrFull(pattern, skeletonizeFull[t_i])&&prev.get(i,j,0)){
                                M.set(true,i,j,0);
                                break;
                            }
                            else{
                                M.set(false, i,j,0);
                            }
                        }
                    }
                }
            
            }
            
            if(opType==1 || opType==2){
                for(int i=0; i<src.height;i++){
                    for(int j=0; j<src.width; j++){
                            
                            bool pattern[9];
                            M.getBitPatternMorphOpBoolFull(i,j,pattern);
                             
                            for(int t_i=0; t_i<38; t_i++){
                                if(matchPatternStageTwoFull(pattern, shrinkThinUnconditionFull[t_i])){
                                    res.set(true,i,j,0);
                                    break;
                                }
                                else{
                                    res.set(false,i,j,0);
                                }
                            }
                    }
                }
                for(int i=0;i<src.height;i++){
                    for(int j=0;j<src.width;j++){
                        G.set(prev.get(i,j,0)&&(!M.get(i,j,0)||res.get(i,j,0)),i,j,0);
                    }
                }
            }
            else if(opType==3){
                for(int i=0; i<src.height;i++){
                    for(int j=0; j<src.width; j++){
                            
                            bool pattern[9];
                            M.getBitPatternMorphOpBoolFull(i,j,pattern);
                             
                            for(int t_i=0; t_i<53; t_i++){
                                if(matchPatternStageTwoFull(pattern, skeletonizeUnconditional[t_i])&&!M.get(i,j,0)){
                                    res.set(true,i,j,0);
                                    break;
                                }
                                else{
                                    res.set(false,i,j,0);
                                }
                            }
                    }
                }
                for(int i=0;i<src.height;i++){
                    for(int j=0;j<src.width;j++){
                        G.set(prev.get(i,j,0)&&(!M.get(i,j,0)||res.get(i,j,0)),i,j,0);
                    }
                }
            }
            cout << "Iteration " << iteration_counter << " " << compareImages(prev,G) << endl;
            iteration_counter++;
            // M.saveImage(("HW3_material/iteration_stageOne_"+to_string(iteration_counter)+".raw").c_str());
            // res.saveImage(("HW3_material/iteration_stageTwo_"+to_string(iteration_counter)+".raw").c_str());
            G.saveImage(("HW3_material/iteration_out_"+to_string(iteration_counter)+".raw").c_str());
            
        }while(!compareImages(prev,G));
        G.saveImage(outFileLocation);
    }
    void performMorphOps(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
        BoolImage src(fileLocation, width, height, bytesPerPixel);
        
        cout << "Shrinking op" << endl;
        applyMorphOp(src, "shrink.raw", 1, false);
        cout << "Thinning op" << endl;
        applyMorphOp(src, "thin.raw", 2, false);
        cout << "Skeletonizing op" << endl;
        applyMorphOp(src, "skeleton.raw", 3, false);

    }
    void solveMaze(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
        BoolImage src(fileLocation, width, height, bytesPerPixel);

        // src.invert();
        applyMorphOp(src, outFileLocation, 1, false);

    }
    void defectDetection(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
        BoolImage src(fileLocation, width, height, bytesPerPixel);

        src.invert();
        applyMorphOp(src,"dots.raw",1, false);
        src.saveImage("dots.raw");
        BoolImage srcCopy(fileLocation, width, height, bytesPerPixel);
        BoolImage dotImage("dots.raw", width, height, bytesPerPixel);
        int count=0;
        for(int i=0; i<dotImage.height;i++){
            for(int j=0; j<dotImage.width; j++){
                if((i==144 && j==360) || (i==293 && j==160) || (i==244 && j==291) || (i==282 && j==224) || (i==287 && j==336))
                    continue;
                if(dotImage.get(i,j,0)){
                    if(count==0){
                        count++;
                        continue;
                    }
                    int size = srcCopy.fillPoint(i,j);
                    cout << size << ",";
                    srcCopy.saveImage(outFileLocation);
                }
            }
        }
        cout << endl;
        srcCopy.saveImage(outFileLocation);
    }
    void CCL(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
        BoolImage src(fileLocation, width, height, bytesPerPixel);
        FloatImage label(width, height, bytesPerPixel);
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                label.set(0,i,j,0);
            }
        }
        int label_id=1;
        
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                int neighbors[4]={0,0,0,0};

                if(src.get(i,j,0)==false){
                    label.set(0,i,j,0);
                    continue;
                }
                else{
                    neighbors[0] = label.get(i-1,j-1,0,false);
                    neighbors[1] = label.get(i-1,j,0,false);
                    neighbors[2] = label.get(i-1,j+1,0,false);
                    neighbors[3] = label.get(i,j-1,0,false);
                    int max = 0;
                    for(int i=0; i<4;i++){
                        if(neighbors[i]>max){
                            max=neighbors[i];
                        }
                    }
                    int min = max;
                    for(int i=0;i<4;i++){
                        // if(neighbors[i]==0)
                            // continue;
                        cout << neighbors[i] << " ";
                        if(neighbors[i]<min && neighbors[i]>0)
                            min = neighbors[i];
                    }
                    cout << min << endl;
                    if(min==0){
                        label.set(label_id++, i, j,0);
                    }
                    else{
                        label.set(min, i,j,0);
                    }
                }
            }
        }
        int class_cnt= 0;
        for(int j=0;j<width;j++){
            for(int i=0;i<height; i++){
                label.spreadLabel(i,j,0);
                if(label.get(i,j,0)>class_cnt){
                    class_cnt = label.get(i,j,0);
                }
            }
        }
        int sizes[class_cnt];
        for(int i=0; i<class_cnt;i++)
            sizes[i]=0;
        for(int i=0;i<height; i++){
            for(int j=0; j<width; j++){
                sizes[(int)label.get(i,j,0)]+=1;
            }
        }
        cout << "Defect Sizes:" << endl;
        for(int i=0; i<class_cnt;i++)
            cout <<sizes[i]<<",";
        label.saveImage(outFileLocation);
        cout << "Labels: " << label_id << " " << class_cnt<< endl;

    }
};



int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: input_image.raw output_image.raw [width] [height] [bytesPerPixel] (0/1/2/3)" << endl;
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

    ProblemThreeSolver solver;
    cout << width << " " << height << " " << bytesPerPixel << argv[1] << " " << argv[2] << endl;
    if(atoi(argv[6])==0){
        solver.performMorphOps(argv[1], width, height, bytesPerPixel, argv[2]);
    }
    else if(atoi(argv[6])==1){
        solver.solveMaze(argv[1], width, height, bytesPerPixel, argv[2]);
    }
    else if(atoi(argv[6])==2){
        solver.defectDetection(argv[1], width, height, bytesPerPixel, argv[2]);
    }
    else{
        solver.CCL(argv[1], width, height, bytesPerPixel, argv[2]);
    }
    
    
}