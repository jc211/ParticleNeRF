#pragma once
#include <neural-graphics-primitives/common.h>

#define PARTICLE_NAMESPACE_BEGIN namespace particle {
#define PARTICLE_NAMESPACE_END }

PARTICLE_NAMESPACE_BEGIN
template<uint32_t POS_DIMS>
using Vectorxf = Eigen::Matrix<float, POS_DIMS, 1>; 

template<typename T, uint32_t POS_DIMS>
using Vector = Eigen::Matrix<T, POS_DIMS, 1>; 

template <typename T, uint32_t DIMS>
__global__ void clip_max_kernel(const uint32_t num_elements, T* __restrict__ inout, float threshold) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;
	float norm = 0.0f;
	#pragma unroll
	for (uint32_t d = 0; d < DIMS; ++d) {
		norm += (float)inout[i * DIMS + d] * (float)inout[i * DIMS + d];
	}
	norm = sqrtf(norm);
	if(norm > threshold) {
		#pragma unroll
		for (uint32_t d = 0; d < DIMS; ++d) {
			inout[i * DIMS + d] = (T)((float)inout[i * DIMS + d] * threshold / norm);
		}
	}
}
PARTICLE_NAMESPACE_END