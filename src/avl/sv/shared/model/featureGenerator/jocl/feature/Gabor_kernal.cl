__kernel void gabor_kernal(	__global uchar *tiles, 
                                __global float *realKernel,
                                __global float *imagKernel,
                                __global float *outGabor,
                                __local float *subSum_Shared,
                                const int4 params){
    const int numOfTiles = params.s0; 
    const int numOfPixels = params.s1; 
    const int tileDim =  params.s2;
    const int kernelDim =  params.s3;
    
    // Variables
    const int imgIndex = get_group_id(0);
    const int t = get_local_id(0); 
    
    if (imgIndex >= numOfTiles){ return; }
    __global const uchar *tile = &tiles[imgIndex*numOfPixels];

    // GenerateCoOccurFeatures
    if (t < (tileDim-kernelDim)){
        float subSum = 0;
        for (int j = 0; j < (tileDim-kernelDim); j++){
            float sR = 0; 
            float sI = 0;
            for (int x = 0; x < kernelDim; x++){
                for (int y = 0; y < kernelDim; y++){
                    int imgIdx = (t+x)+(j+y)*tileDim;
                    int kernelIdx = x+y*kernelDim;
                    sR += tile[imgIdx]*realKernel[kernelIdx];
                    sI += tile[imgIdx]*imagKernel[kernelIdx];
                }
            }
            subSum += sqrt(sR*sR+sI*sI);
        }
        subSum_Shared[t] = subSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // CollectFeatures
    if (t == 0){
        float sum = 0;
        for (int j = 0; j < tileDim-kernelDim; j++){
            sum += subSum_Shared[j];
        }
        outGabor[imgIndex] = sum/(tileDim*tileDim); 
    }
    
}


