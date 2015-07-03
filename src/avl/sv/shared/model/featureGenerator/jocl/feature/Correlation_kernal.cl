__kernel void correlation_kernal(   __global uchar *tiles,
                                    __global float *means,
                                    __global float *moment2s,
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
    float mean = means[imgIndex];
    float moment2 = moment2s[imgIndex];

    // GenerateCoOccurFeatures
    if (t < tileDim){
        float sumD = 0,	sumR = 0;

        for (int j = 0; j< tileDim; j++){
            int index = (j+0)+(t+0)*tileDim;
            float pC = tile[index];                    
            if (j < (tileDim-1)){ 
                float pR = tile[index+1];
                sumR += ((float)((pC-mean)*(pR-mean))*normFact);
            }
            if (t < (tileDim-1)){ 
                float pD = tile[index+tileDim];
                sumD += ((float)((pC-mean)*(pD-mean))*normFact);
            }
        }
        subSumD_Shared[t] = sumD;
        subSumR_Shared[t] = sumR;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // CollectFeatures
    if (t == 0){
        float sumD = 0, sumR = 0;
        for (int j = 0; j< tileDim; j++){
            sumD += subSumD_Shared[j];
            sumR += subSumR_Shared[j];
        }
        outR[imgIndex] = sumR/moment2; 
        outD[imgIndex] = sumD/moment2; 
    }	
    
}
