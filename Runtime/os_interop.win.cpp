#ifdef _WIN32
#include "os_interop.hpp"
#include <windows.h>
#include <iostream>

kernel_launcher get_launcher_by_name(
	const char* name
) {
	std::cout << "trying to load " << name << std::endl;
	HMODULE current_module = GetModuleHandleA(NULL);
	kernel_launcher launcher = (kernel_launcher) GetProcAddress(current_module, name);

	return launcher;
}

uint64_t measure_time_nanos()
{
	LARGE_INTEGER perf, freq;

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&perf);

	double time = (uint64_t)perf.QuadPart;
	time /= freq.QuadPart;
	// Time in seconds

	return (uint64_t)(time * 1e9);
}

#endif
