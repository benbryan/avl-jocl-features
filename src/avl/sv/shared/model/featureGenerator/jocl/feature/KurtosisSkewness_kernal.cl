__kernel void kurtosisSkewness_kernal(   __global uchar *tiles,
                                    __global float *means,
                                    __global float *kurtosisOut,
                                    __global float *skewnessOut,
                                    const int4 params){
    const int numOfTiles = params.s0; 
    const int numOfPixels = params.s1; 
    const int tileDim =  params.s2;
        
    // Variables
    const int imgIndex = get_group_id(0);
    const int t = get_local_id(0); 
    
    if (imgIndex >= numOfTiles){ return; }
    __global const uchar *tile = &tiles[imgIndex*numOfPixels];

    __local float   pShared[256];
    __local int     histShared[256];

    // Initialization
    if (t==0)  { 
        for (int i = 0; i < 256; i++){
            histShared[i] = 0; 
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Generate Histogram
    if (t < tileDim){
        for (int j = 0; j < tileDim; j++){
            int index = j+t*tileDim;
            float pC = tile[index];
            atomic_add(&histShared[tile[index]],1);
        }       
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 256; i++){
        pShared[i] = ((float)histShared[i])/numOfPixels; 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // GenerateMoments
    if (t == 0){
        float mean = means[imgIndex];
        float m2 = 0, m3 = 0, m4 = 0;
        for (int i = 0; i < 256; i++){
            float temp = (i-mean)*(i-mean)*pShared[i];
            m2 += temp;
            temp *= (i-mean);
            m3 += temp;
            temp *= (i-mean);
            m4 += temp;
        }
        kurtosisOut[imgIndex] = m4/(m2*m2); 
        skewnessOut[imgIndex] = m3/ pow(m2, (float)1.5);
    }    
}

