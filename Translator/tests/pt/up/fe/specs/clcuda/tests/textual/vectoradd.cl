kernel void vectoradd(global int* A, global float* B, global int* C) {
	size_t i = get_global_id(0);
	
	C[i] = A[i] + (int) B[i];
}
