kernel void vectoradd(global int* A, global int* B, global int* C) {
	size_t i = get_global_id(0);
	
	C[i] = A[i] + B[i];
}
