# cl-in-cuda

A (very basic) OpenCL-to-CUDA compiler/runtime API.
Requires access to the OpenCL code at compile-time, and the generated CUDA must be linked into the final program.

Supports only a very limited subset of OpenCL excluding e.g. textures but including fine-grained shared virtual memory.

**NOTE**: This project is **NOT** actively maintained.
