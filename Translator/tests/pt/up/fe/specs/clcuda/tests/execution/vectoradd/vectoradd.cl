typedef int DATATYPE;

__kernel void vectoradd(global DATATYPE* out, global DATATYPE* A, global DATATYPE* B)
{
    size_t i = get_global_id(0);

    out[i] = A[i] + B[i];
}
