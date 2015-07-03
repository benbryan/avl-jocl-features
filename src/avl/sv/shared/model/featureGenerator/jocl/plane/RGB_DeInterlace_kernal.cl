__kernel void rgb_DeInterlace_kernal( __global uchar *tiles, 
                            __global uchar *outRs,
                            __global uchar *outGs,
                            __global uchar *outBs,
                            const int2 params ){
    // input interlaced RGB
    // output seperated RGB means 
    const int bands = 3;
    const int numOfTiles = params.s0;
    const int numOfPixels = params.s1;

    const int imgIndex = get_group_id(0);
    const int t = get_local_id(0);
    const int tileDim = get_local_size(0);
    
    if ((imgIndex < numOfTiles) && (t < tileDim)){  
        __global const uchar *tile = &tiles[imgIndex*numOfPixels*3];
        __global uchar *outR = &outRs[imgIndex*numOfPixels];
        __global uchar *outG = &outGs[imgIndex*numOfPixels];
        __global uchar *outB = &outBs[imgIndex*numOfPixels];

        const int index = t*tileDim;
        for (int i = 0; i < tileDim; i++){
            outR[ index+i ] = tile[ (index +i)*3 + 0 ];
            outG[ index+i ] = tile[ (index +i)*3 + 1 ];
            outB[ index+i ] = tile[ (index +i)*3 + 2 ];
        }
    }
}