__kernel void homogeneity_kernal(  __global uchar *tiles, 
                                   __global float *outR,
                                   __global float *outD,
                                   __local float *subSumD_Shared,
                                   __local float *subSumR_Shared,
                                   const int4 params){
    const int numOfTiles = params.s0; 
    const int numOfPixels = params.s1; 
    const int tileDim =  params.s2;
    const float normFact = 1.0/((float)numOfPixels-(float)tileDim);
    
    // Variables
    const int imgIndex = get_group_id(0);
    const int t = get_local_id(0); 
    
    if (imgIndex >= numOfTiles){ return; }
    __global const uchar *tile = &tiles[imgIndex*numOfPixels];

    if (t < tileDim){
        float sumD = 0, sumR = 0;
        for (int j = 0; j< tileDim; j++){
            int index = t*tileDim+j;
            float pC = tile[index];                    
            if (j < (tileDim-1)){ 
                float pR = tile[index+1];
                sumR += (normFact/(1.0+fabs(pC-pR)));
            }
            if (t < (tileDim-1)){ 
                float pD = tile[index+tileDim]; 
                sumD += (normFact/(1.0+fabs(pC-pD)));
            }
        }
        subSumD_Shared[t] = sumD;
        subSumR_Shared[t] = sumR;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (t == 0){
        float sumD = 0, sumR = 0;
        for (int j = 0; j< tileDim; j++){
            sumD += subSumD_Shared[j];
            sumR += subSumR_Shared[j];
        }
        outR[imgIndex] = sumR; 
        outD[imgIndex] = sumD; 
    }	
}
