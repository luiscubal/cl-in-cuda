#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define CHECK(functionName, ...) validate(__FILE__, __LINE__, #functionName, functionName(__VA_ARGS__))
#define CHECK_CODE(functionName, errorCode) validate(__FILE__, __LINE__, #functionName, errorCode)

void validate(const char *file, int line, const char *functionName, cl_int err) {
    if (err != CL_SUCCESS) {
        printf("Error %d at %s:%d (%s)", (int) err, file, line, functionName);
        exit(1);
    }
}

#define N 1024

int main() {
    srand(time(NULL));

    cl_platform_id platform;
    CHECK(clGetPlatformIDs, 1, &platform, NULL);

    cl_device_id device;
    CHECK(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    cl_int errcode;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode);
	CHECK_CODE(clCreateContext, errcode);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode);
	CHECK_CODE(clCreateCommandQueue, errcode);

    uint8_t *B1 = (uint8_t*) calloc(N, sizeof(uint8_t));
    uint16_t *B2 = (uint16_t*) calloc(N, sizeof(uint16_t));
    uint32_t *B4 = (uint32_t*) calloc(N, sizeof(uint32_t));
	uint64_t *B8 = (uint64_t*) calloc(N, sizeof(uint64_t));

    cl_mem B1_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
    cl_mem B2_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint16_t) * N, NULL, &errcode);
    CHECK_CODE(clCreateBuffer, errcode);
	cl_mem B4_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * N, NULL, &errcode);
	CHECK_CODE(clCreateBuffer, errcode);
	cl_mem B8_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t) * N, NULL, &errcode);
	CHECK_CODE(clCreateBuffer, errcode);

	uint8_t placeholder1 = 0xCD;
	uint16_t placeholder2 = 0xCDCD;
	uint32_t placeholder4 = 0xCDCDCDCD;
	uint64_t placeholder8 = UINT64_C(0xCDCDCDCDCDCDCDCD);
	// Initialize buffers
	CHECK(clEnqueueFillBuffer, queue, B1_gpu, (void*) &placeholder1, 1, 0, sizeof(uint8_t) * N, 0, NULL, NULL);
	CHECK(clEnqueueFillBuffer, queue, B2_gpu, (void*) &placeholder2, 2, 0, sizeof(uint16_t) * N, 0, NULL, NULL);
	CHECK(clEnqueueFillBuffer, queue, B4_gpu, (void*) &placeholder4, 4, 0, sizeof(uint32_t) * N, 0, NULL, NULL);
	CHECK(clEnqueueFillBuffer, queue, B8_gpu, (void*) &placeholder8, 8, 0, sizeof(uint64_t) * N, 0, NULL, NULL);
	CHECK(clFinish, queue);

	uint8_t pat1 = 0x99U;
    CHECK(clEnqueueFillBuffer, queue, B1_gpu, (void*) &pat1, 1, 4, 800, 0, NULL, NULL);
	uint16_t pat2 = 0x9897U;
	CHECK(clEnqueueFillBuffer, queue, B2_gpu, (void*) &pat2, 2, 4, 800, 0, NULL, NULL);
	uint32_t pat4 = 0x98979695U;
	CHECK(clEnqueueFillBuffer, queue, B4_gpu, (void*) &pat4, 4, 4, 800, 0, NULL, NULL);
	uint64_t pat8 = UINT64_C(0x9897969594939291);
	CHECK(clEnqueueFillBuffer, queue, B8_gpu, (void*) &pat8, 8, 8, 800, 0, NULL, NULL);
	CHECK(clFinish, queue);

    CHECK(clEnqueueReadBuffer, queue, B1_gpu, CL_TRUE, 0, sizeof(uint8_t) * N, B1, 0, NULL, NULL);
	CHECK(clEnqueueReadBuffer, queue, B2_gpu, CL_TRUE, 0, sizeof(uint16_t) * N, B2, 0, NULL, NULL);
	CHECK(clEnqueueReadBuffer, queue, B4_gpu, CL_TRUE, 0, sizeof(uint32_t) * N, B4, 0, NULL, NULL);
	CHECK(clEnqueueReadBuffer, queue, B8_gpu, CL_TRUE, 0, sizeof(uint64_t) * N, B8, 0, NULL, NULL);

#define ASSERT_EQUALS(exp, act) if ((exp) != (act)) {\
	printf("Failed assertion at %d. Expected %x, got %x", __LINE__, (int)(exp), (int)(act));\
	return 1;\
}

	ASSERT_EQUALS(placeholder1, B1[3]);
	ASSERT_EQUALS(pat1, B1[4]);
	ASSERT_EQUALS(pat1, B1[803]);
	ASSERT_EQUALS(placeholder1, B1[804]);
	ASSERT_EQUALS(placeholder2, B2[1]);
	ASSERT_EQUALS(pat2, B2[2]);
	ASSERT_EQUALS(pat2, B2[401]);
	ASSERT_EQUALS(placeholder2, B2[402]);
	ASSERT_EQUALS(placeholder4, B4[0]);
	ASSERT_EQUALS(pat4, B4[1]);
	ASSERT_EQUALS(pat4, B4[200]);
	ASSERT_EQUALS(placeholder4, B4[201]);
	ASSERT_EQUALS(placeholder8, B8[0]);
	ASSERT_EQUALS(pat8, B8[1]);
	ASSERT_EQUALS(pat8, B8[100]);
	ASSERT_EQUALS(placeholder8, B8[101]);
	
	CHECK(clReleaseMemObject, B1_gpu);
	CHECK(clReleaseMemObject, B2_gpu);
	CHECK(clReleaseMemObject, B4_gpu);
	CHECK(clReleaseMemObject, B8_gpu);

    printf("It works\n");
    return 0;
}
