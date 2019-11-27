/*
Some comment
*/
// Single line
kernel void comment(global int* A, global int* B, global int* C) {
	size_t i = get_global_id(0);
	
	// Body
	/* Block body */
	C[i] = A[i] + B[i];
}
