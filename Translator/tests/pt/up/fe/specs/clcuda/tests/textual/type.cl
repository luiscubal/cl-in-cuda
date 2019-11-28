typedef size_t mytype;

kernel void type(global int* A, global float* B, global int* C) {
	mytype i = get_global_id(0);
	
	C[i] = A[i] + (int) B[i];
}
