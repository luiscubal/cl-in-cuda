__kernel void manualbound(global float* A, int N) {
	size_t global_id;
	
	global_id = get_global_id(0U);
	if (global_id < N) {
		A[global_id] = 1.0;
	}
}
