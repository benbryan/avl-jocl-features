__kernel void rgb_DeInterlaceToHSV_kernal( __global uchar *tiles, 
                            __global uchar *outHs,
                            __global uchar *outSs,
                            __global uchar *outVs,
                            const int2 params ){
    // input interlaced RGB
    // output seperated HSV means 
    const int bands = 3;
    const int numOfTiles = params.s0;
    const int numOfPixels = params.s1;

    const int imgIndex = get_group_id(0);
    const int t = get_local_id(0);
    const int tileDim = get_local_size(0);
    
    if ((imgIndex < numOfTiles) && (t < tileDim)){  
        __global const uchar *tile = &tiles[imgIndex*numOfPixels*3];
        __global uchar *outH = &outHs[imgIndex*numOfPixels];
        __global uchar *outS = &outSs[imgIndex*numOfPixels];
        __global uchar *outV = &outVs[imgIndex*numOfPixels];

        const int index = t*tileDim;
        for (int i = 0; i < tileDim; i++){
            float r = (float)tile[ (index +i)*3 + 0 ]/255;
            float g = (float)tile[ (index +i)*3 + 1 ]/255;
            float b = (float)tile[ (index +i)*3 + 2 ]/255;
            float h, s, v;
            
            float min = fmin(fmin(r, g), b);
            float max = fmax(fmax(r, g), b);
            float delta = max - min;
            v = max;				// v
            if( max != 0 ) {
                s = delta / max;		// s
            } else {
                // r = g = b = 0		// s = 0, v is undefined
                s = 0;
                h = -1;
            }
            if( r == max ) {
                h = ( g - b ) / delta;		// between yellow & magenta
            } 
            if( g == max ) {
                h = 2 + ( b - r ) / delta;	// between cyan & yellow
            }
            if( b == max ) {
                h = 4 + ( r - g ) / delta;	// between magenta & cyan
            }
            
            h *= 60;				// degrees
            if ( h < 0 ) {
                h += 360;
            }         
            
            outH[ index+i ] = h/(float)360*(float)255;
            outS[ index+i ] = s*255;
            outV[ index+i ] = v*255;
        }
    }
}
