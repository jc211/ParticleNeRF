#pragma once

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>

#include <vector>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <particles/common.h>
#include <particles/search_structure.cuh>
#include <particles/pbd.cuh>
#include <particles/rbf.cuh>
#include <particles/optimizers/adam.h>
namespace cg = cooperative_groups;

PARTICLE_NAMESPACE_BEGIN

using json = nlohmann::json;

template<typename T, uint32_t POS_DIMS>
struct ParticleData {
    tcnn::GPUMemory<uint32_t> ids;
    tcnn::GPUMemory<Vectorxf<POS_DIMS>> velocities;

    tcnn::GPUMemory<Vectorxf<POS_DIMS>> positions;
    tcnn::GPUMemory<float> features;

    tcnn::GPUMemory<Vector<T, POS_DIMS>> positions_gradient;
    tcnn::GPUMemory<T> features_gradient;

	std::shared_ptr<tcnn::Optimizer<T>> position_optimizer;
	std::shared_ptr<tcnn::Optimizer<T>> feature_optimizer;

    void initialize_optimizers(const json& config) {
	    position_optimizer.reset(new particle::AdamOptimizer<T>(config.value("position_optimizer", json())));
	    feature_optimizer.reset(new particle::AdamOptimizer<T>(config.value("feature_optimizer", json())));
    }

    void resize(uint32_t n_points, uint32_t dim_features) {
        ids.resize(n_points);
        velocities.resize(n_points);
        velocities.memset(0);

        positions.resize(n_points);
        features.resize(n_points * dim_features);

        positions_gradient.resize(n_points);
        features_gradient.resize(n_points * dim_features);

		position_optimizer->allocate(n_points*POS_DIMS);
		feature_optimizer->allocate(n_points*dim_features);
    }
};

template<uint32_t POS_DIMS>
__global__ void generate_2d_particles_grid_kernel(uint32_t n_points, Eigen::Vector2i res_2d, Vectorxf<POS_DIMS>* __restrict__ out) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x>=res_2d.x() || y>=res_2d.y())
		return;
	uint32_t i = x+ y*res_2d.x();
    if(i >= n_points) return;
	Eigen::Vector2f pos = Eigen::Vector2f{(float)x, (float)y}.cwiseQuotient((res_2d-Eigen::Vector2i::Ones()).cast<float>());
	out[i] = pos; 
}

template<uint32_t POS_DIMS>
__global__ void generate_3d_particles_grid_kernel(uint32_t n_points, Eigen::Vector3i res_3d, Vectorxf<POS_DIMS>* __restrict__ out) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3d.x() || y>=res_3d.y() || z>=res_3d.z())
		return;
	uint32_t i = x+ y*res_3d.x() + z*res_3d.x()*res_3d.y();
    if(i >= n_points) return;
	Eigen::Vector3f pos = Eigen::Vector3f{(float)x, (float)y, (float)z}.cwiseQuotient((res_3d-Eigen::Vector3i::Ones()).cast<float>());
	out[i] = pos; 
}

template<uint32_t POS_DIMS>
inline __device__ float norm(const Vectorxf<POS_DIMS>& a, const Vectorxf<POS_DIMS>& b) {
    return (a-b).norm();
}

template<uint32_t POS_DIMS>
inline __device__ Vectorxf<POS_DIMS> norm_derivative(const Vectorxf<POS_DIMS>& a, const Vectorxf<POS_DIMS>& b) {
    float norm =  (a-b).norm();
    if(norm < 1e-15f) return Vectorxf<POS_DIMS>::Zero();
	return 0.5f / norm * (a-b);
}

template<uint32_t POS_DIMS>
__global__ void out_of_bounds_kernel(
    size_t num_particles,
    tcnn::MatrixView<float> position,
    uint32_t* __restrict__ alive)
{
    // https://github.com/NVIDIA/cuda-samples/blob/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_particles)
        return;
    Vectorxf<POS_DIMS> p;
    #pragma unroll
    for(int dim=0; dim<POS_DIMS; dim++) {
        p[dim] = position(dim, i);
        if(p[dim] < 0 || p[dim] > 1) {
            alive[i] = 0;
            return;
        }
    }
}

template<typename T, uint32_t POS_DIMS>
__global__ void sort_by_index(
    uint32_t num_particles,
    const uint32_t* __restrict__ particle_index,
    int dim_features,

    uint32_t* __restrict__ ids,
    const Vectorxf<POS_DIMS>* __restrict__ velocities,
    const Vectorxf<POS_DIMS>* __restrict__ positions,
    const float* __restrict__ features,

    const Vector<T, POS_DIMS>* __restrict__ position_gradients,
    const T* __restrict__ feature_gradients,

    uint32_t* __restrict__ sorted_ids,
    Vectorxf<POS_DIMS>* __restrict__ sorted_velocities,
    Vectorxf<POS_DIMS>* __restrict__ sorted_positions,
    float* __restrict__ sorted_features,

    Vector<T, POS_DIMS>* __restrict__ sorted_position_gradients,
    T* __restrict__ sorted_feature_gradients
    )
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_particles)
        return;

    uint32_t sorted_index = particle_index[i];
    sorted_ids[i] = ids[sorted_index];
    sorted_velocities[i] = velocities[sorted_index];
    sorted_positions[i] = positions[sorted_index];
    #pragma unroll
    for(int dim =0 ; dim < dim_features; ++dim) {
        sorted_features[i * dim_features + dim] = features[sorted_index * dim_features + dim];
        sorted_feature_gradients[i * dim_features + dim] = feature_gradients[sorted_index * dim_features + dim];
    }
    sorted_position_gradients[i] = position_gradients[sorted_index];
}

template<typename T>
__global__ void sort_adam_optimizer(
    uint32_t num_particles,
    const uint32_t* __restrict__ particle_index,
    int element_dim,

    const float* __restrict__ first_moments,
    const float* __restrict__ second_moments,
	const uint32_t* __restrict__ param_steps,

    float* __restrict__ sorted_first_moments,
    float* __restrict__ sorted_second_moments,
	uint32_t* __restrict__ sorted_param_steps
    )
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_particles)
        return;
    uint32_t sorted_index = particle_index[i];
    #pragma unroll
    for(int dim =0 ; dim < element_dim; ++dim) {
        sorted_first_moments[i * element_dim + dim] = first_moments[sorted_index * element_dim + dim];
        sorted_second_moments[i * element_dim + dim] = second_moments[sorted_index * element_dim + dim];
        sorted_param_steps[i * element_dim + dim] = param_steps[sorted_index * element_dim + dim];
    }
}

template<typename T, uint32_t POS_DIMS>
__global__ void compact_particles_kernel(
    uint32_t num_particles,
    const uint32_t* __restrict__ alive,
    const uint32_t* __restrict__ rank,

    uint32_t dim_features,

    uint32_t* num_compacted,
    const uint32_t* __restrict__ ids,
    const Vectorxf<POS_DIMS>* __restrict__ positions,
    const Vectorxf<POS_DIMS>* __restrict__ velocities,
    const float* __restrict__ features,

    const Vector<T, POS_DIMS>* __restrict__ position_gradients,
    const T* __restrict__ feature_gradients,

    uint32_t* __restrict__ sorted_ids,
    Vectorxf<POS_DIMS>* __restrict__ sorted_positions,
    Vectorxf<POS_DIMS>* __restrict__ sorted_velocities,
    float* __restrict__ sorted_features,

    Vector<T, POS_DIMS>* __restrict__ sorted_position_gradients,
    T* __restrict__ sorted_feature_gradients
    )
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_particles)
        return;

    if(i == num_particles - 1) {
        *num_compacted = alive[i] ? rank[i] + 1 : rank[i];
    }
    
    if(alive[i] == 0)
        return;

    uint32_t compact_index = rank[i];
    sorted_ids[compact_index] = ids[i];
    sorted_positions[compact_index] = positions[i];
    sorted_velocities[compact_index] = velocities[i];
    sorted_position_gradients[compact_index] = position_gradients[i];
    #pragma unroll
    for(int dim =0 ; dim < dim_features; ++dim) {
        sorted_features[compact_index * dim_features + dim] = features[i * dim_features + dim];
        sorted_feature_gradients[compact_index * dim_features + dim] = feature_gradients[i * dim_features + dim];
    }
}

template<typename T>
__global__ void create_id_to_index(
	uint32_t n_particles,
	const uint32_t* __restrict__ index_to_id,
	uint32_t* __restrict__ id_to_index
)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_particles)
        return;

    uint32_t id = index_to_id[i];
	id_to_index[id] = i;
}

template<typename T>
void sort_optimizer(cudaStream_t stream, uint32_t n_samples, uint32_t dim, const uint32_t* sort_key, tcnn::Optimizer<T>* optimizer, tcnn::Optimizer<T>* sorted) {
    if(auto* optim = dynamic_cast<particle::AdamOptimizer<T>*>(optimizer)) {
        auto* sorted_optim = dynamic_cast<particle::AdamOptimizer<T>*>(sorted);
        tcnn::linear_kernel(
            sort_adam_optimizer<T>,
            0, stream,
            n_samples, 
            sort_key,
            dim,
            optim->first_moments().data(),
            optim->second_moments().data(),
            optim->param_steps().data(),
            sorted_optim->first_moments().data(),
            sorted_optim->second_moments().data(),
            sorted_optim->param_steps().data()
        ); 
    } else {
        throw std::runtime_error("Unsupported optimizer");
    }  
}


struct ParticleInfo {
    bool valid = false;
    void* grid_info_gpu;
    void* grid_info;
	float* positions;
	uint32_t* cell_start;
	uint32_t* cell_end;
};


template<typename T, uint32_t POS_DIMS>
class Particles {
    public:
    Particles(
        uint32_t n_points, 
        uint32_t dim_features,
        const json& config);

    void resize(uint32_t n_points);
    void shrink_to_fit();
    void init_random(tcnn::default_rng_t &rng, uint32_t n_points, uint32_t offset = 0);
    void init_positions_grid(tcnn::default_rng_t &rng, uint32_t n_points);

    void update(cudaStream_t stream, float search_radius = 1.0f);
    void sort(cudaStream_t stream, uint32_t* sort_key);

    void remove_particles(cudaStream_t stream, uint32_t* alive_mask);
    void remove_out_of_bounds(cudaStream_t stream);
    void add_particles(cudaStream_t stream, uint32_t n_points, Vectorxf<POS_DIMS>* positions, float* features);
    void physics_step(
        cudaStream_t stream,
        uint32_t n_steps,
        float nerf_scale,
        float constraint_softness,
        float min_distance,
        float velocity_damping,
        float timestep,
        bool use_collisions
        );

    uint32_t size() const { return m_n_points; }
    ParticleData<T, POS_DIMS>& get_particle_data() { return m_data[m_current_buffer]; }
    ParticleData<T, POS_DIMS>& get_buffered_particle_data() { return m_data[(m_current_buffer+1) % 2]; }
    void swap_buffers() { m_current_buffer = (m_current_buffer + 1) % 2; }
    uint32_t* get_ids() { return get_particle_data().ids.data();}
    Vectorxf<POS_DIMS>* get_position() { return get_particle_data().positions.data();}
    float* get_features() { return get_particle_data().features.data();}
    uint32_t get_feature_dim() { return m_dim_features;}
    std::shared_ptr<tcnn::Optimizer<T>> get_position_optimizer() { return get_particle_data().position_optimizer; }
    std::shared_ptr<tcnn::Optimizer<T>> get_feature_optimizer() { return get_particle_data().feature_optimizer; }
    ParticleInfo get_particle_info() { 
        const auto& search_structure = m_particle_search_structure;
        // Fix this grid_info gpu
        return {true, nullptr, (void*)&search_structure.grid_info, (float*)get_position(), search_structure.cell_start.data(), search_structure.cell_end.data()}; 
    }

    Vector<T, POS_DIMS>* get_position_gradient() { return get_particle_data().positions_gradient.data();}
    T* get_feature_gradient() { return get_particle_data().features_gradient.data();}

    uint32_t m_n_points;
    uint32_t m_capacity;
    uint32_t m_dim_features;
    float m_last_search_radius = 0.1f;

    ParticleData<T, POS_DIMS> m_data[2];
    uint32_t m_current_buffer = 0;

    SearchStructure<POS_DIMS> m_particle_search_structure;
    uint32_t m_update_id = 1;

};

template<typename T, uint32_t POS_DIMS>
Particles<T, POS_DIMS>::Particles(uint32_t n_points, uint32_t dim_features, const json& config): m_capacity(0), m_n_points(0),  m_dim_features(dim_features) {
    get_particle_data().initialize_optimizers(config);
    get_buffered_particle_data().initialize_optimizers(config);
    resize(n_points);
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::init_random(tcnn::default_rng_t &rng, uint32_t n_points, uint32_t offset) {
    if(m_n_points == 0) return;
    auto& particle_data = get_particle_data();
	// tcnn::generate_random_uniform<float>(rng, n_points*POS_DIMS, (float*)particle_data.positions.data() + POS_DIMS*offset, 0.0f, 1.0f);
    init_positions_grid(rng, n_points);
	tcnn::generate_random_uniform<float>(rng, n_points*m_dim_features, (float*)particle_data.features.data() + m_dim_features*offset, -1e-2f, 1e-2f);
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::init_positions_grid(tcnn::default_rng_t &rng, uint32_t n_points) {
    if(m_n_points == 0) return;
    auto& particle_data = get_particle_data();
    if(POS_DIMS == 3) {
        // resolution of grid 
        uint32_t res = std::ceil(std::pow(n_points, 1.0f/3.0f));
        const dim3 threads = { 16, 8, 1 };
        const dim3 blocks = { tcnn::div_round_up(res, threads.x), tcnn::div_round_up(res, threads.y), tcnn::div_round_up(res, threads.z) };
        Eigen::Vector3i res3d = {res, res, res};
	    generate_3d_particles_grid_kernel<3><<<blocks, threads, 0, 0>>>(n_points, res3d, (Vectorxf<3>*)particle_data.positions.data()); 

    } else if(POS_DIMS == 2) {
        uint32_t res = std::ceil(std::pow(n_points, 1.0f/2.0f));
        const dim3 threads = { 16, 8, 1};
        const dim3 blocks = { tcnn::div_round_up(res, threads.x), tcnn::div_round_up(res, threads.y), 1  };
        Eigen::Vector2i res2d = {res, res};
	    generate_2d_particles_grid_kernel<2><<<blocks, threads, 0, 0>>>(n_points, res2d, (Vectorxf<2>*)particle_data.positions.data()); 

    } else {
        throw std::runtime_error("Only 2D and 3D grids are supported");
    }
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::resize(uint32_t n_points) {
    auto& particle_data = get_particle_data();
    particle_data.resize(n_points, m_dim_features);
    thrust::sequence(thrust::cuda::par.on(0), particle_data.ids.data(), particle_data.ids.data() + n_points);
    auto& auxilliary_buffer = get_buffered_particle_data();
    auxilliary_buffer.resize(n_points, m_dim_features);
    m_n_points = n_points;
    m_capacity = n_points;
}


template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::sort(cudaStream_t stream, uint32_t* sort_key) {
    auto& particle_data = get_particle_data();
    auto& sorted_data = get_buffered_particle_data();
    uint32_t num_particles = m_n_points;

    tcnn::linear_kernel(
        sort_by_index<T, POS_DIMS>,
        0, stream,
        num_particles, 
        sort_key,
        m_dim_features,

        particle_data.ids.data(),
        particle_data.velocities.data(),

        particle_data.positions.data(),
        particle_data.features.data(),

        particle_data.positions_gradient.data(),
        particle_data.features_gradient.data(),

        sorted_data.ids.data(),
        sorted_data.velocities.data(),

        sorted_data.positions.data(),
        sorted_data.features.data(),

        sorted_data.positions_gradient.data(),
        sorted_data.features_gradient.data()
    ); 

    sort_optimizer<T>(stream, num_particles, POS_DIMS,  sort_key, particle_data.position_optimizer.get(), sorted_data.position_optimizer.get()); //Not needed if physics is ON
    sort_optimizer<T>(stream, num_particles, m_dim_features, sort_key, particle_data.feature_optimizer.get(), sorted_data.feature_optimizer.get());
    swap_buffers();
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::update(cudaStream_t stream, float search_radius) {
    uint32_t num_particles = m_n_points;
    if(search_radius < 0.0f) {
        search_radius = m_last_search_radius;
    }
    m_last_search_radius = search_radius;
    if(num_particles == 0)
        return;

    // Update Search Structure
    auto& particle_data = get_particle_data();
    auto& search_structure = m_particle_search_structure;
    tcnn::GPUMatrixDynamic<float> positions((float*)particle_data.positions.data(), POS_DIMS, num_particles);
    search_structure.update(
        positions, 
        2.0*search_radius, 
        Vectorxf<POS_DIMS>::Constant(0), 
        Vectorxf<POS_DIMS>::Constant(1), 
        stream);

    // Sort by Search Structure
    uint32_t* particle_index = search_structure.sorted_to_unsorted_index.data();
    sort(stream, particle_index);

    // Update id counter for visualization
    m_update_id++;
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::add_particles(cudaStream_t stream, uint32_t num_particles, Vectorxf<POS_DIMS>* positions, float* features) {
    tcnn::default_rng_t rnd;
    uint32_t n_old_points = m_n_points;
    resize(m_n_points + num_particles);
    auto& particle_data = get_particle_data();
    cudaMemcpyAsync(particle_data.positions.data()+n_old_points, positions, num_particles*sizeof(Vectorxf<POS_DIMS>), cudaMemcpyDeviceToDevice, stream);
    if(features == nullptr) {
	    tcnn::generate_random_uniform<float>(rnd, num_particles*m_dim_features, (float*)particle_data.features.data() + m_dim_features*n_old_points, -1e-2f, 1e-2f);
    } else {
        cudaMemcpyAsync(particle_data.features.data()+n_old_points*m_dim_features, features, num_particles*m_dim_features, cudaMemcpyDeviceToDevice, stream);
    }
    update(stream, m_last_search_radius);
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::remove_particles(cudaStream_t stream, uint32_t* alive_mask) {
    tcnn::GPUMemoryArena::Allocation alloc;
    auto scratch = tcnn::allocate_workspace_and_distribute<
        uint32_t, // n_compacted_points
        uint32_t // rank
    >(stream, &alloc, 
        1,
        m_n_points
    ); 
    uint32_t* n_compacted_points = std::get<0>(scratch);
    uint32_t* rank = std::get<1>(scratch);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), alive_mask, alive_mask+m_n_points, rank);
    auto& particle_data = get_particle_data();
    auto& sorted_data = get_buffered_particle_data();
    tcnn::linear_kernel(
        compact_particles_kernel<T, POS_DIMS>,
        0, stream,
        m_n_points, 
        alive_mask,
        rank,
        m_dim_features,
        n_compacted_points,

        particle_data.ids.data(),
        particle_data.positions.data(),
        particle_data.velocities.data(),
        particle_data.features.data(),

        particle_data.positions_gradient.data(),
        particle_data.features_gradient.data(),

        sorted_data.ids.data(),
        sorted_data.positions.data(),
        sorted_data.velocities.data(),
        sorted_data.features.data(),

        sorted_data.positions_gradient.data(),
        sorted_data.features_gradient.data()
    ); 
    swap_buffers();
    cudaMemcpyAsync(&m_n_points, n_compacted_points, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    update(stream, m_last_search_radius);
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::remove_out_of_bounds(cudaStream_t stream) {
    auto& particle_data = get_particle_data();

    tcnn::GPUMatrixDynamic<float> positions((float*)particle_data.positions.data(), POS_DIMS, m_n_points);
    tcnn::GPUMemoryArena::Allocation alloc;
    auto scratch = tcnn::allocate_workspace_and_distribute<
        uint32_t           // alive 
    >(stream, &alloc, m_n_points); 

    uint32_t* mask = std::get<0>(scratch);
	tcnn::GPUMatrix<uint32_t> alive_mask(mask, 1, m_n_points);
	thrust::fill(thrust::cuda::par.on(stream), alive_mask.data(), alive_mask.data()+m_n_points, 1);
    tcnn::linear_kernel(
        out_of_bounds_kernel<POS_DIMS>,
        0, stream,
        m_n_points, 
        positions.view(),
        alive_mask.data()
    ); 
    remove_particles(stream, mask);
}

template<typename T, uint32_t POS_DIMS>
void Particles<T, POS_DIMS>::physics_step(
    cudaStream_t stream,
    uint32_t n_steps,
    float nerf_scale,
    float constraint_softness,
    float min_distance,
    float velocity_damping,
    float timestep,
    bool use_collisions
    ) {
    if(n_steps == 0) 
        return;
    uint32_t n_particles = m_n_points; 

    auto& particle_data = get_particle_data();
    auto& sorted_data = get_buffered_particle_data();
    const auto& search_struct = m_particle_search_structure;

    float ts = timestep / (float)n_steps;
    for(int i = 0; i < n_steps; i++) {
        tcnn::linear_kernel(
            physics_integrator_kernel<T, POS_DIMS>,
            0, stream,
            n_particles, 
            ts,
            nerf_scale/n_steps,
            velocity_damping,
            particle_data.positions.data(),
            particle_data.velocities.data(),
            particle_data.positions_gradient.data(),
            sorted_data.positions.data()
        );

        if(use_collisions && min_distance > 0) {
            tcnn::linear_kernel(
                physics_constraints_kernel<T, POS_DIMS>,
                0, stream,
                n_particles, 
                ts,
                constraint_softness,
                min_distance,
                search_struct.grid_info,
                particle_data.positions.data(),
                search_struct.cell_start.data(),
                search_struct.cell_end.data(),
                sorted_data.positions.data()
            );

            tcnn::linear_kernel(
                update_velocities_kernel<T, POS_DIMS>,
                0, stream,
                n_particles, 
                ts,
                particle_data.positions.data(),
                sorted_data.positions.data(),
                particle_data.velocities.data()
            );
        }
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        std::swap(particle_data.positions, sorted_data.positions);
    }
    cudaMemsetAsync((void*)particle_data.positions_gradient.data(), 0, n_particles*POS_DIMS*sizeof(T), stream);
	remove_out_of_bounds(stream);
}


PARTICLE_NAMESPACE_END