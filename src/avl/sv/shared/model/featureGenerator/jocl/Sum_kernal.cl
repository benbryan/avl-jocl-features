__kernel void sum_kernal( __global float *f1, 
                                __global float *f2,
                                __global float *fout,
                                const int2 params ){
    const int numOfTiles = params.s0;
    const int imgIndex = get_global_id(0);
    
    if ((imgIndex < numOfTiles)){  
        fout[imgIndex] = f1[imgIndex]+f2[imgIndex];
    }
}