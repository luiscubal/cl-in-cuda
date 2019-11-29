#define TS 16

typedef int DATATYPE;

__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global DATATYPE* A,
                      const __global DATATYPE* B,
                      __global DATATYPE* C) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    DATATYPE acc = 0;
    for (int k = 0; k < K; ++k) {
    	acc += A[k * M + globalRow] * B[globalCol * K + k];
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}

__kernel void myGEMM2(const int M, const int N, const int K,
                      const __global DATATYPE* A,
                      const __global DATATYPE* B,
                      __global DATATYPE* C,
                      __local DATATYPE* Asub,
                      __local DATATYPE* Bsub) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col * TS + row] = A[tiledCol*M + globalRow];
        Bsub[col * TS + row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k * TS + row] * Bsub[col * TS + k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}
