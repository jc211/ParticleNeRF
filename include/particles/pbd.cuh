#pragma once
#include <particles/common.h>
#include <particles/search_structure.cuh>

PARTICLE_NAMESPACE_BEGIN

template<typename T, uint32_t POS_DIMS>
__global__ void physics_integrator_kernel(
	uint32_t num_points,
    float time_step,
    float scale,
    float damping,
	const Vectorxf<POS_DIMS>* __restrict__ positions,
	Vector<float, POS_DIMS>* __restrict__ velocities,
	const Vector<T, POS_DIMS>* __restrict__ gradients,
	Vectorxf<POS_DIMS>* __restrict__ new_positions
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;
    velocities[query_index] = damping*velocities[query_index] - gradients[query_index].template cast<float>() * scale; 
    new_positions[query_index] = positions[query_index] + velocities[query_index] * time_step;
}

template<typename T, uint32_t POS_DIMS>
__global__ void update_velocities_kernel(
	uint32_t num_points,
    float time_step,
	const Vectorxf<POS_DIMS>* __restrict__ prev_positions,
	const Vectorxf<POS_DIMS>* __restrict__ current_positions,
	Vector<float, POS_DIMS>* __restrict__ velocities
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;
    velocities[query_index] = (current_positions[query_index] - prev_positions[query_index]).template cast<float>() / (time_step );
}

template<typename T, uint32_t POS_DIMS>
__global__ void physics_constraints_kernel(
	uint32_t num_points,
    float time_step,
    float alpha,
    float min_distance,
	const GridInfo<POS_DIMS> grid_info,
	const Vectorxf<POS_DIMS>* __restrict__ particles,
	const uint32_t* __restrict__ cell_start,
	const uint32_t* __restrict__ cell_end,
	Vectorxf<POS_DIMS>* __restrict__ new_positions
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;
    const Vectorxf<POS_DIMS>& query = particles[query_index];
    const Vectorxf<POS_DIMS>& current_pos = new_positions[query_index];
    const Vectorxf<POS_DIMS>& prev_pos = query; 

    float search_radius = grid_info.cell_size * 0.5f;

    NEIGHBOUR_SEARCH({
        const Vectorxf<POS_DIMS>& current_neighbour_position = new_positions[neighbour_index];
        const Vectorxf<POS_DIMS>& prev_neighbour_position = particles[neighbour_index];
        const Vectorxf<POS_DIMS> current_delta = current_pos - current_neighbour_position;
        const Vectorxf<POS_DIMS> prev_delta = prev_pos - prev_neighbour_position;
        float current_dist = current_delta.norm();
        float prev_dist = prev_delta.norm();
        float diff = prev_dist - current_dist;
        
        if(current_dist < min_distance && current_dist > 1e-9f) {
            Vectorxf<POS_DIMS> correction = 0.5f*current_delta.normalized() * (min_distance - current_dist);
            #pragma unroll
            for(int dim = 0; dim < POS_DIMS; ++dim) {
                atomicAdd((float*)&current_pos + dim, correction[dim]);
            }
        } 
    });
}

PARTICLE_NAMESPACE_END 