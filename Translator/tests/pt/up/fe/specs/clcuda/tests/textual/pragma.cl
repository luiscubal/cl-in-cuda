#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel void pragma(global double* A, global double* B, global double* C) {
	size_t i = get_global_id(0);
	
	C[i] = A[i] + B[i];
}
