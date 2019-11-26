#ifdef _WIN32
#include "cl_interface_shared.h"
#include <windows.h>
#include <iostream>

kernel_launcher get_launcher_by_name(const char* name) {
	std::cout << "trying to load " << name << std::endl;
	HMODULE current_module = GetModuleHandleA(NULL);
	kernel_launcher launcher = (kernel_launcher) GetProcAddress(current_module, name);

	return launcher;
}

#endif
