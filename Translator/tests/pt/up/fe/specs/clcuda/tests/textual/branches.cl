kernel void branches(global int* A, global int* B) {
	size_t i = get_global_id(0);
	
	if (A[i] > 0) {
		B[i] = A[i];
	} else {
		B[i] = -A[i];
	}
}
