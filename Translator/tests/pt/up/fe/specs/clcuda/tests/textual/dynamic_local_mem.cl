kernel void dynamic_local_mem(local int* A, local double* B) {
	A[get_local_id(0)] = B[get_local_id(0)];
}
