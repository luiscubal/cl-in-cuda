typedef float DATATYPE;

__kernel void reduce_add_float(global DATATYPE* out, global DATATYPE* A)
{
    size_t i = get_global_id(0);

    out[i] = work_group_reduce_add(A[i]);
}
