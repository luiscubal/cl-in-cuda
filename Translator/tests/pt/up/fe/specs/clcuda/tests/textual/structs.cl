struct array {
	global float* data;
	size_t dim1;
};

kernel void structs(global float* data, int dim1) {
	struct array arr;
	
	arr.data = data;
	arr.dim1 = dim1;
	
	arr.data[arr.dim1 - get_global_id(0) - 1] = get_global_id(0);
}
