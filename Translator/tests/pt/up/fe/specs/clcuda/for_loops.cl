kernel void for_loops(global int* A, int N) {
	for (int i = 0; i < N; i++) {
		A[N * get_global_size(0) + get_global_id(0)] = 1;
	}
}
