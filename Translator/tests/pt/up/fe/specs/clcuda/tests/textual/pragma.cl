#if __OPENCL_VERSION__ < 120
// After OpenCL 1.2, using this extension becomes a warning
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel void pragma(global double* A, global double* B, global double* C) {
	size_t i = get_global_id(0);
	
	C[i] = A[i] + B[i];
}
