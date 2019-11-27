kernel void modifiers(global const int* restrict A, global const int* restrict B, global int* restrict C) {
	size_t i = get_global_id(0);
	
	C[i] = A[i] + B[i];
}
