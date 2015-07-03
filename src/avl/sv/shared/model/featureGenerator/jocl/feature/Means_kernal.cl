__kernel void means_kernal( __global uchar *tiles, 
                            __global float *means,
                            const int2 params ){
    // input interlaced RGB
    // output seperated RGB means 
    const int numOfTiles = params.s0;
    const int numOfPixels = params.s1;

    const int imgIndex = get_global_id(0);
    __global const uchar *tile = &tiles[imgIndex*numOfPixels];

    if (imgIndex >= numOfTiles){ return; }

    float temp = 0;
    for (int i = 0; i < numOfPixels; i++){
        temp += tile[i];
    }
    means[imgIndex] = (float)temp/(float)numOfPixels;
}