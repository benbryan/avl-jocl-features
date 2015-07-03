__kernel void rgb_ToGray_kernal( __global uchar *tiles, 
                            __global uchar *grays,
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
        __global uchar *gray = &grays[imgIndex*numOfPixels];

        const int index = t*tileDim;
        for (int i = 0; i < tileDim; i++){
            gray[ index+i ] =   0.29893601f * tile[ (index+i)*3 + 0 ] + 
                                0.58704305f * tile[ (index+i)*3 + 1 ] + 
                                0.11402091f * tile[ (index+i)*3 + 2 ];
        }
    }
}