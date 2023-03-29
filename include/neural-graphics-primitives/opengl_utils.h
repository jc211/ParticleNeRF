#pragma once
#include <neural-graphics-primitives/common.h>

NGP_NAMESPACE_BEGIN
#ifdef NGP_GUI
void glCheckError(const char* file, unsigned int line);
bool check_shader(uint32_t handle, const char* desc, bool program);
uint32_t compile_shader(bool pixel, const char* code);
#endif //NGP_GUI
NGP_NAMESPACE_END