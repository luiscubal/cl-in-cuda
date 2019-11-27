kernel void while_loop(global int* A, int a) {
	size_t i = get_global_id(0);
	
	while (A[i] < a) {
		A[i] *= 2;
	}
}
