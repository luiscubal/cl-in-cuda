#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(functionName, ...) validate(__FILE__, __LINE__, #functionName, functionName(__VA_ARGS__))
#define CHECK_CODE(functionName, errorCode) validate(__FILE__, __LINE__, #functionName, errorCode)

void validate(const char *file, int line, const char *functionName, cl_int err) {
    if (err != CL_SUCCESS) {
        printf("Error %d at %s:%d (%s)", (int) err, file, line, functionName);
        exit(1);
    }
}

static cl_program create_program_from_file(cl_context context, cl_device_id device, const char *filename, const char *options) {
	FILE* f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Could not open OpenCL file.\n");
		abort();
	}
	fseek(f, 0, SEEK_END);
	size_t length = (size_t) ftell(f);
	fseek(f, 0, SEEK_SET);
    char *content = (char*) malloc(length);
	if (!content) {
		fprintf(stderr, "Could not allocate memory.\n");
		abort();
	}
	if (fread(content, 1, length, f) != length) {
		fprintf(stderr, "Failed to read OpenCL file.\n");
		abort();
	}
	fclose(f);
	
    cl_int errcode;
	cl_program program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char**) &content, NULL, &errcode);
	CHECK_CODE(clCreateProgramWithSource, errcode);
	
	errcode = clBuildProgram(program, 1, &device, options, NULL, NULL);
	
	size_t log_size;
	CHECK(clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	
	if (log_size > 1) {
		char *log = (char*) malloc(log_size);
		CHECK(clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		
		fprintf(stderr, "Program log: %s\n", log);
	} else {
		fprintf(stderr, "Program compiled without errors or warnings.\n");
	}
	
	CHECK_CODE(clBuildProgram, errcode);
	
	return program;
}

#ifndef LOCAL_SIZE
#error "Undefined LOCAL_SIZE constant. Use -DLOCAL_SIZE=<N>"
#endif

typedef cl_float DATATYPE;
#define N 2048

int main() {
    srand(time(NULL));

    cl_platform_id platform;
    CHECK(clGetPlatformIDs, 1, &platform, NULL);

    cl_device_id device;
    CHECK(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    cl_int errcode;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode);
	CHECK_CODE(clCreateContext, errcode);

    cl_program program = create_program_from_file(context, device, "reduce_add_float.cl.toml", "");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode);
	CHECK_CODE(clCreateCommandQueue, errcode);

    cl_kernel reduce_add_float = clCreateKernel(program, "reduce_add_float", &errcode);
	CHECK_CODE(clCreateKernel, errcode);

    DATATYPE *A = (DATATYPE*) malloc(sizeof(DATATYPE) * N);
    DATATYPE *out = (DATATYPE*) malloc(sizeof(DATATYPE) * N);

    for (int i = 0; i < N; i++) {
        // Good randomness is not that important. rand() works just fine
        A[i] = (float)(rand() % 100);
    }

    cl_mem A_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(DATATYPE) * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
    cl_mem out_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DATATYPE) * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);

    CHECK(clEnqueueWriteBuffer, queue, A_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N, A, 0, NULL, NULL);

    size_t local_work_size = LOCAL_SIZE;
    size_t global_work_size = N;

    CHECK(clSetKernelArg, reduce_add_float, 0, sizeof(cl_mem), (void*) &out_gpu);
    CHECK(clSetKernelArg, reduce_add_float, 1, sizeof(cl_mem), (void*) &A_gpu);
    CHECK(clEnqueueNDRangeKernel, queue, reduce_add_float, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    CHECK(clFinish, queue);

    CHECK(clEnqueueReadBuffer, queue, out_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N, out, 0, NULL, NULL);

    for (int i = 0; i < N; i += LOCAL_SIZE) {
        float local_result = 0;
        for (int j = 0; j < LOCAL_SIZE; ++j) {
            // While floating point addition is not always associative, it should be for
            // numbers with few significant bits.
            local_result += A[i + j];
        }
        for (int j = 0; j < LOCAL_SIZE; ++j) {
            if (out[i] != local_result) {
                printf("Error at element %d. Expected %f, got %f\n", i, local_result, out[i]);
                return 1;
            }
        }
    }

    printf("It works\n");
    return 0;
}
