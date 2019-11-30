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
	CHECK_CODE(clCreateProgramWithBinary, errcode);
	
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

typedef cl_int DATATYPE;
#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16
#define N 128

int main() {
    srand(time(NULL));

    cl_platform_id platform;
    CHECK(clGetPlatformIDs, 1, &platform, NULL);

    cl_device_id device;
    CHECK(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    cl_int errcode;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode);
	CHECK_CODE(clCreateContext, errcode);

    cl_program program = create_program_from_file(context, device, "mygemm2.cl.toml", "");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode);
	CHECK_CODE(clCreateCommandQueue, errcode);

    cl_kernel mygemm1 = clCreateKernel(program, "myGEMM1", &errcode);
	CHECK_CODE(clCreateKernel, errcode);
    cl_kernel mygemm2 = clCreateKernel(program, "myGEMM2", &errcode);
	CHECK_CODE(clCreateKernel, errcode);

    DATATYPE *A = (DATATYPE*) malloc(sizeof(DATATYPE) * N * N);
    DATATYPE *B = (DATATYPE*) malloc(sizeof(DATATYPE) * N * N);
    DATATYPE *out1 = (DATATYPE*) malloc(sizeof(DATATYPE) * N * N);
    DATATYPE *out2 = (DATATYPE*) malloc(sizeof(DATATYPE) * N * N);
	
    for (int i = 0; i < N * N; i++) {
        // Good randomness is not that important. rand() works just fine
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    cl_mem A_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(DATATYPE) * N * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
    cl_mem B_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(DATATYPE) * N * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
    cl_mem out1_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DATATYPE) * N * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
    cl_mem out2_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(DATATYPE) * N * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);

    CHECK(clEnqueueWriteBuffer, queue, A_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N * N, A, 0, NULL, NULL);
    CHECK(clEnqueueWriteBuffer, queue, B_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N * N, B, 0, NULL, NULL);

    size_t local_work_size[] = { LOCAL_SIZE_X, LOCAL_SIZE_Y };
    size_t global_work_size[] = { N, N };

	cl_int size = N;
    CHECK(clSetKernelArg, mygemm1, 0, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm1, 1, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm1, 2, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm1, 3, sizeof(cl_mem), (void*) &A_gpu);
    CHECK(clSetKernelArg, mygemm1, 4, sizeof(cl_mem), (void*) &B_gpu);
    CHECK(clSetKernelArg, mygemm1, 5, sizeof(cl_mem), (void*) &out1_gpu);
    CHECK(clEnqueueNDRangeKernel, queue, mygemm1, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK(clFinish, queue);
    
    CHECK(clSetKernelArg, mygemm2, 0, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm2, 1, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm2, 2, sizeof(cl_int), (cl_int*) &size);
    CHECK(clSetKernelArg, mygemm2, 3, sizeof(cl_mem), (void*) &A_gpu);
    CHECK(clSetKernelArg, mygemm2, 4, sizeof(cl_mem), (void*) &B_gpu);
    CHECK(clSetKernelArg, mygemm2, 5, sizeof(cl_mem), (void*) &out2_gpu);
    CHECK(clSetKernelArg, mygemm2, 6, sizeof(DATATYPE) * LOCAL_SIZE_X * LOCAL_SIZE_Y, NULL);
    CHECK(clSetKernelArg, mygemm2, 7, sizeof(DATATYPE) * LOCAL_SIZE_X * LOCAL_SIZE_Y, NULL);
    CHECK(clEnqueueNDRangeKernel, queue, mygemm2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK(clFinish, queue);
	
    CHECK(clEnqueueReadBuffer, queue, out1_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N * N, out1, 0, NULL, NULL);
    CHECK(clEnqueueReadBuffer, queue, out2_gpu, CL_TRUE, 0, sizeof(DATATYPE) * N * N, out2, 0, NULL, NULL);

	printf("After read back\n");
	DATATYPE* ref = malloc(sizeof(DATATYPE) * N * N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			DATATYPE acc = 0;
			for (int k=0; k < N; k++) {
				acc += A[k * N + i] * B[j * N + k];
			}
			ref[j * N + i] = acc;
		}
	}
	printf("After reference generated\n");

    for (int i = 0; i < N * N; i++) {
        if (out1[i] != ref[i]) {
            printf("[1] Error at element %d. Expected %d, got %d\n", i, (int) (ref[i]), (int) out1[i]);
            return 1;
        }
    }
    for (int i = 0; i < N * N; i++) {
        if (out2[i] != ref[i]) {
            printf("[2] Error at element %d. Expected %d, got %d\n", i, (int) (ref[i]), (int) out2[i]);
            return 1;
        }
    }

    printf("It works\n");
    return 0;
}
