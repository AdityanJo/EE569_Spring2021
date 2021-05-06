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
        bool compareImages(Image im_a, Image im_b){
            bool diffFlag=true;
            for(int i=0; i<im_a.height; i++){
                for(int j=0; j<im_a.width; j++){
                    
                    if(im_a.get(i,j,0,false)!=im_b.get(i,j,0,false)){
                        diffFlag=false;
                    }
                }
            }
            return diffFlag;
        }
        bool matchPattern(string src, string dst){
            return strcmp(src.c_str(), dst.c_str());
        }
        bool matchPatternStageTwo(string src, string lookup){
            bool a_u_b_u_c = false;
            bool match = false;
            
            for(int i=0; i<8;i++){
                 
                if(lookup[i]=='M' && src[i]=='1'){
                    // cout << match << " " << lookup[i] << " " << src[i] << endl;
                    match = true;
                }
                if(lookup[i]=='A' || lookup[i]=='B' || lookup[i]=='C'){
                    a_u_b_u_c = a_u_b_u_c || src[i];
                }
                
            }
            if(lookup.find('A')!=-1 || lookup.find('B')!=-1 || lookup.find('C')!=-1){
                
                return a_u_b_u_c && match;
            }   
            else{
                return match;
            }
        }
        void applySkinMorphOp(Image src, char* outFileLocation, int opType=1){
            Image M(src.width, src.height, src.bytesPerPixel);
            Image G(src.width, src.height, src.bytesPerPixel);
            Image P(src.width, src.height, src.bytesPerPixel);
            Image prev(src.width, src.height, src.bytesPerPixel);
            prev.copyFrom(src);

            string shrinkPatterns[59]={
                "01000000", 
                "00010000", 
                "00000100", 
                "00000001",
                "10000000", 
                "00100000", 
                "00001000", 
                "00000010",
                "11000000", 
                "01100000", 
                "00110000", 
                "00001100", 
                "00001100", 
                "00000110", 
                "00000011", 
                "10000001",
                "11000001", 
                "01110000", 
                "00011100", 
                "00000111",
                "10110000", 
                "10100001", 
                "01101000", 
                "11000010", 
                "11100000", 
                "00111000", 
                "00001110", 
                "10000011",
                "10110001", 
                "01101100", 
                "11110000", 
                "11100001", 
                "01111000", 
                "00111100", 
                "00011110", 
                "00001111", 
                "10000111", 
                "11000011",
                "11110001", 
                "01111100", 
                "00011111", 
                "11000111",
                "11100011", 
                "11111000", 
                "00111110", 
                "10001111",
                "11110011", 
                "11100111", 
                "11111100", 
                "11111001", 
                "01111110", 
                "00111111", 
                "10011111", 
                "11001111",
                "11110111", 
                "11111101", 
                "01111111", 
                "11011111"
            };

            string skinBondOne[4] = {
                "01000000", 
                "00010000", 
                "00000100", 
                "00000001"
                };
            string skinBondTwo[4] = {
                "10000000", 
                "00100000", 
                "00001000", 
                "00000010"
                };
            string skinBondThree[8] = {
                "11000000", 
                "01100000", 
                "00110000", 
                "00001100", 
                "00001100", 
                "00000110", 
                "00000011", 
                "10000001",
                };
            string thinSkeletonBondFour[8] = {
                "10100000", 
                "00101000", 
                "00001010", 
                "10000010", 
                "11000001", 
                "01110000", 
                "00011100", 
                "00000111"
                };
            string skinBondFour[4] = {
                "11000001", 
                "01110000", 
                "00011100", 
                "00000111"
                };
            string skinThinFive[8] = {
                "10110000", 
                "10100001", 
                "01101000", 
                "11000010", 
                "11100000", 
                "00111000", 
                "00001110", 
                "10000011"
                };
            string skinThinSix[10] = {
                "10110001", 
                "01101100", 
                "11110000", 
                "11100001", 
                "01111000", 
                "00111100", 
                "00011110", 
                "00001111", 
                "10000111", 
                "11000011"
                };
            string skeletonSix[8] = {
                "11110000", 
                "11100001", 
                "01111000", 
                "00111100", 
                "00011110", 
                "00001111", 
                "10000111", 
                "11000011"
                };
            string skinThinSkeletonSeven[4]={
                "11110001", 
                "01111100", 
                "00011111", 
                "11000111"
                };
            string skinThinSkeletonEight[4] = {
                "11100011", 
                "11111000", 
                "00111110", 
                "10001111"
                };
            string skinThinSkeletonNine[8] = {
                "11110011", 
                "11100111", 
                "11111100", 
                "11111001", 
                "01111110", 
                "00111111", 
                "10011111", 
                "11001111"
                };
            string skinThinSkeletonTen[4] = {
                "11110111", 
                "11111101", 
                "01111111", 
                "11011111"
                };
            string skeletonEleven[4] = {
                "11111011", 
                "11111110", 
                "10111111", 
                "11101111"
                };


            string shrinkThinUncondition[37]={
                //Spur
                "0M000000", "000M0000",
                //Single 4
                "000000M0", "M0000000",
                //L Cluster
                "MM000000", "0MM00000", "00MM0000", "000MM000", "0000MM00", "00000MM0", "000000MM", "M000000M",
                //4-connected offset
                "0MM0M000", "M0MM0000", "M0M0000M", "MM0000M0",
                //Spur corner cluster
                "BMA00M00", "00BMA00M", "0M00AMB0", "B00M00AM",
                //Corner Cluster
                "DDMMMDDD",
                //Tee branch
                "M0MDMD00", "MDM0M00D", "MD00M0MD", "M00DMDM0",
                "0DMDM0M0", "00M0MDMD", "M0M00DMD", "MDMD00M0",
                //Vee branch
                "DMDMDABC", "BCDMDMDA", "DABCDMDM", "DMDABCDM",
                //Diagonal branch
                "M0MD0M0D", "0DM0MD0M", "0M0DM0MD", "MD0M0DM0"
            };

            string skeletonizeUncondition[26]={
                //Spur
                "0000000M", "00000M00", "0M000000", "000M0000",
                //Single 4 connection
                "000000M0", "M0000000", "0000M000", "00M00000",
                //L corner
                "M0M00000", "00M0M000", "M00000M0", "0000M0M0",
                //Corner cluster
                "DDMMMDDD", "MDDDDDMM",
                //Tee branch
                "MDMDMDDD", "DDMDMDMD", "MDDDMDMD", "MDMDDDMD",
                //Vee branch
                "DMDMDABC", "BCDMDMDA", "DABCDMDM", "DMDABCDM",
                //Diagonal branch
                "M0MD0M0D", "0DM0MD0M", "0M0DM0MD", "MD0M0DM0"
            };
            
            int iteration_counter=0;
            do{
                if(iteration_counter>1)
                    prev.copyFrom(G);

                for(int i=0; i<src.height; i++){
                    for(int j=0; j<src.width; j++){
                        if(prev.get(i,j,0,false)==0)
                            continue;
                        string pattern="xxxxxxxx";
                        pattern=prev.getBitPatternMorphOp(i,j,pattern);
                        int center=prev.get(i,j,0,false);
                        int ss_neigh = 0, sc_neigh=0;

                        if(center==1){
                            // cout << pattern;
                            if(pattern[0]=='1')
                                ss_neigh+=1;
                            if(pattern[2]=='1')
                                ss_neigh+=1;
                            if(pattern[4]=='1')
                                ss_neigh+=1;
                            if(pattern[6]=='1')
                                ss_neigh+=1;
                            if(pattern[1]=='1')
                                sc_neigh+=1;
                            if(pattern[3]=='1')
                                sc_neigh+=1;
                            if(pattern[5]=='1')
                                sc_neigh+=1;
                            if(pattern[7]=='1')
                                sc_neigh+=1;
                        }
                        else{
                            if(pattern[0]=='0')
                                ss_neigh+=1;
                            if(pattern[2]=='0')
                                ss_neigh+=1;
                            if(pattern[4]=='0')
                                ss_neigh+=1;
                            if(pattern[6]=='0')
                                ss_neigh+=1;
                            if(pattern[1]=='0')
                                sc_neigh+=1;
                            if(pattern[3]=='0')
                                sc_neigh+=1;
                            if(pattern[5]=='0')
                                sc_neigh+=1;
                            if(pattern[7]=='0')
                                sc_neigh+=1;
                        }
                        int bond_number = 2*ss_neigh + sc_neigh;
                        // cout << pattern << " " << bond_number << endl ;
                        if(opType==1){
                            bool status=false;
                            for(int t_i=0; t_i<59;t_i++){
                                if(matchPattern(pattern, shrinkPatterns[t_i]))
                                    status=true;
                            }
                            if(status==true){
                                M.set(1, i, j, 0);
                            }
                            else
                            {
                                M.set(0, i,j,0);
                            }
                            // if(bond_number==1){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinBondOne[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==2){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinBondTwo[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==3){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<8;t_i++){
                            //         if(matchPattern(pattern, skinBondThree[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==4){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinBondFour[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==5){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<8;t_i++){
                            //         if(matchPattern(pattern, skinThinFive[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==6){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<8;t_i++){
                            //         if(matchPattern(pattern, skinThinSix[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==7){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinThinSkeletonSeven[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==8){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinThinSkeletonEight[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==9){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<8;t_i++){
                            //         if(matchPattern(pattern, skinThinSkeletonNine[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                            // else if(bond_number==10){
                            //     bool status=false;
                            //     for(int t_i=0; t_i<4;t_i++){
                            //         if(matchPattern(pattern, skinThinSkeletonTen[t_i]))
                            //             status=true;
                            //     }
                            //     if(status==true){
                            //         M.set(1, i, j, 0);
                            //     }
                            //     else
                            //     {
                            //         M.set(0, i,j,0);
                            //     }
                            // }
                        }
                        else if(opType==2){
                            if(bond_number==4){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, thinSkeletonBondFour[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==5){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinFive[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==6){
                                bool status=false;
                                for(int t_i=0; t_i<10;t_i++){
                                    if(matchPattern(pattern, skinThinSix[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==7){
                                bool status=false;
                                for(int t_i=0; t_i<4;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonSeven[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==8){
                                bool status=false;
                                for(int t_i=0; t_i<4;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonEight[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==9){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonNine[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==10){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonTen[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                        }
                        else if(opType==3){
                            if(bond_number==4){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, thinSkeletonBondFour[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==6){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skeletonSix[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==7){
                                bool status=false;
                                for(int t_i=0; t_i<4;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonSeven[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==8){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonEight[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==9){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonNine[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==10){
                                bool status=false;
                                for(int t_i=0; t_i<8;t_i++){
                                    if(matchPattern(pattern, skinThinSkeletonTen[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                            else if(bond_number==11){
                                bool status=false;
                                for(int t_i=0; t_i<4;t_i++){
                                    if(matchPattern(pattern, skeletonEleven[t_i]))
                                        status=true;
                                }
                                if(status==true){
                                    M.set(1, i, j, 0);
                                }
                                else
                                {
                                    M.set(0, i,j,0);
                                }
                            }
                        }                        

                    }
                }
                if(opType==1 || opType==2){
                            
                    for(int i=0;i<src.height; i++){
                        for(int j=0; j<src.width; j++){
                            if(M.get(i,j,0,false)==0){
                                G.set(prev.get(i,j,0),i,j,0);
                            }
                            else{
                                string pattern="xxxxxxxx";
                                pattern= prev.getBitPatternMorphOp(i,j,pattern);
                                for(int t_i=0; t_i<37; t_i++){
                                    if(matchPatternStageTwo(pattern,shrinkThinUncondition[t_i])){
                                        G.set(0, i,j, 0);
                                    }
                                
                                }
                                
                            }
                        }
                    }
                }
                else{
                    for(int i=0;i<src.height; i++){
                        for(int j=0; j<src.width; j++){
                            if(M.get(i,j,0,false)==0){
                                G.set(prev.get(i,j,0),i,j,0);
                            }
                            else{
                                string pattern="xxxxxxxx";
                                pattern= prev.getBitPatternMorphOp(i,j,pattern);
                                for(int t_i=0; t_i<26; t_i++){
                                    if(matchPatternStageTwo(pattern,skeletonizeUncondition[t_i])){
                                        G.set(0, i,j, 0);
                                    }
                                    else{
                                        G.set(prev.get(i,j,0, false), i,j, 0);
                                    }
                                }
                                
                            }
                        }
                    }
                }
                iteration_counter++;
                if(iteration_counter>100)
                    break;
                // cout << compareImages(prev,G) << " images comparison " << endl;
                if (opType==3){
                    for(int i=0;i<src.height; i++){
                        for(int j=0; j<src.width; j++){
                            string pattern="xxxxxxxx";
                            pattern = G.getBitPatternMorphOp(i,j,pattern);
                            int center=prev.get(i,j,0,false);
                            int ss_neigh = 0, sc_neigh=0;

                            if(center==1){
                                // cout << pattern;
                                if(pattern[0]=='1')
                                    ss_neigh+=1;
                                if(pattern[2]=='1')
                                    ss_neigh+=1;
                                if(pattern[4]=='1')
                                    ss_neigh+=1;
                                if(pattern[6]=='1')
                                    ss_neigh+=1;
                                if(pattern[1]=='1')
                                    sc_neigh+=1;
                                if(pattern[3]=='1')
                                    sc_neigh+=1;
                                if(pattern[5]=='1')
                                    sc_neigh+=1;
                                if(pattern[7]=='1')
                                    sc_neigh+=1;
                            }
                            else{
                                if(pattern[0]=='0')
                                    ss_neigh+=1;
                                if(pattern[2]=='0')
                                    ss_neigh+=1;
                                if(pattern[4]=='0')
                                    ss_neigh+=1;
                                if(pattern[6]=='0')
                                    ss_neigh+=1;
                                if(pattern[1]=='0')
                                    sc_neigh+=1;
                                if(pattern[3]=='0')
                                    sc_neigh+=1;
                                if(pattern[5]=='0')
                                    sc_neigh+=1;
                                if(pattern[7]=='0')
                                    sc_neigh+=1;
                            }
                            int bond_number = 2*ss_neigh + sc_neigh;
                            if(bond_number>0 && G.get(i,j,0)!=0){
                                G.set(1,i,j,0);
                            }
                        }
                    }
                }
                src.copyFrom(M);
                src.multiply(255);
                src.saveImage(("HW3_material/iteration_"+to_string(iteration_counter)+".raw").c_str());
                
            }while(!compareImages(prev,G));
            // M.multiply(255);
            // M.saveImage(outFileLocation);
            // getchar();
            G.multiply(255);
            G.saveImage(outFileLocation);

            
        }
        void performMorphOps(char *fileLocation, int width, int height, int bytesPerPixel, char *outFileLocation){
            Image src(fileLocation, width, height, bytesPerPixel);
            src.binarize(127, 1);
            applySkinMorphOp(src, outFileLocation, 1);

        }
};

int main(int argc, char *argv[]){
    int width, height, bytesPerPixel;

    if(argc<3){
        cout << "Usage wrong: (0/1/2) input_image.raw output_image.raw [width] [height] [bytesPerPixel]" << endl;
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
    solver.performMorphOps(argv[1], width, height, bytesPerPixel, argv[2]);
    
}