#pragma once
#include <particles/common.h>

PARTICLE_NAMESPACE_BEGIN
enum class RBFType: int {
	Guassian,
	Linear,
    Bump
};
static constexpr const char* RBFTypeStr = "Guassian\0Linear\0Bump\0\0";

inline __device__ float bump_rbf(float x, float search_radius) {
    if(abs(x) >= (search_radius - 1e-5f)) return 0.0f;
    float search_radius_squared = search_radius * search_radius;
    const float x2 = 1.0f/(1.0f-x*x/search_radius_squared);
    return __expf(-x2);

}

inline __device__ float bump_rbf_derivative(float x, float search_radius) {
    if(abs(x) >= (search_radius - 1e-5f)) return 0.0f;
    float search_radius_squared = search_radius * search_radius;
    const float temp = (-2*search_radius_squared*x)/((search_radius_squared-x*x)*(search_radius_squared-x*x));
    return temp*bump_rbf(x, search_radius); 

}

inline __device__ float guassian_rbf(float x, float search_radius_squared) {
	const float s = -6/search_radius_squared;
	const float y = s*x*x;
	return __expf(y);
}

inline __device__ float guassian_rbf_derivative(float x, float search_radius_squared) {
	const float s = -6/search_radius_squared;
	const float y = s*x*x;
	return 2*s*x*__expf(y);
}

inline __device__ float linear_rbf(float x, float search_radius) {
	const float slope = x > 0 ? -1.0f/search_radius : 1.0/search_radius;
	return 1 + slope*x;
}

inline __device__ float linear_rbf_derivative(float x, float search_radius) {
	const float slope = x > 0 ? -1.0f/search_radius : 1.0/search_radius;
	return slope;
}

inline __device__ float rbf(float x, float radius, RBFType rkf_type) {
	switch(rkf_type) {
		case RBFType::Guassian:
			return guassian_rbf(x, radius*radius);
		case RBFType::Linear:
			return linear_rbf(x, radius);
		case RBFType::Bump:
			return bump_rbf(x, radius);
	}
	return 0.0f;
}

inline __device__ float rbf_derivative(float x, float radius, RBFType rkf_type) {
	switch(rkf_type) {
		case RBFType::Guassian:
			return guassian_rbf_derivative(x, radius*radius);
		case RBFType::Linear:
			return linear_rbf_derivative(x, radius);
		case RBFType::Bump:
			return bump_rbf_derivative(x, radius);
	}
	return 0.0f;
}

PARTICLE_NAMESPACE_END 