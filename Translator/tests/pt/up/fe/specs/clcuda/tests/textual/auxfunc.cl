int apply(int x) {
	return x * 2;
}

kernel void auxfunc(global int* A, global int* B) {
	size_t i = get_global_id(0);
	
	A[i] = apply(B[i]);
}
