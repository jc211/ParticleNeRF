#pragma once
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <particles/common.h>

namespace cg = cooperative_groups;

PARTICLE_NAMESPACE_BEGIN


// Probably should have used a lambda here
#define NEIGHBOUR_SEARCH(do_stuff) do {\
    Vector<uint32_t, POS_DIMS> coord = grid_pos<POS_DIMS>(query, grid_info.min, grid_info.cell_size);\
	for (int z = -1; z <= 1; z++) {\
		for (int y = -1; y <= 1; y++){\
			for (int x = -1; x <= 1; x++) {\
				Vector<uint32_t, 3> offset(x, y, z);\
				Vector<uint32_t, POS_DIMS> cell_index = coord;\
				for(int dim=0; dim<POS_DIMS;dim++) {\
					cell_index[dim] += offset[dim];\
					if(cell_index[dim] < 0 || cell_index[dim] >= grid_info.dim[dim]) {\
						goto next_cell;\
					}\
				}\
                uint32_t hash = hash_grid_cell<POS_DIMS>(grid_info, cell_index);\
				uint32_t neighbour_cell_start = cell_start[hash];\
                if (neighbour_cell_start != 0xffffffff) {\
                    uint32_t neighbour_cell_end = cell_end[hash];\
                    for (uint32_t neighbour_index = neighbour_cell_start; neighbour_index < neighbour_cell_end; neighbour_index++) {\
                        do_stuff\
                    }\
                }\
            }\
			next_cell:\
			continue;\
        }\
		if(POS_DIMS == 2) break;\
	}\
}while(0)

template<uint32_t POS_DIMS>
__global__ void compute_min_max_kernel(
	uint32_t num_particles,
	tcnn::MatrixView<const float> point,
	float cell_size,
	uint32_t* min_cell,
	uint32_t* max_cell
)
{
	uint particle_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (particle_index >= num_particles) return;

    #pragma unroll
    for(int dim = 0; dim<POS_DIMS; dim++) {
        uint32_t cell = (uint32_t)(point(dim, particle_index) / cell_size);
	    atomicMin(&(min_cell[dim]), cell);
	    atomicMax(&(max_cell[dim]), cell);
    }
}


template<uint32_t POS_DIMS>
void compute_min_max(const tcnn::GPUMatrixDynamic<float>& points, float cell_size, Vectorxf<POS_DIMS>& min_point, Vectorxf<POS_DIMS>& max_point) {
        tcnn::GPUMemory<Vector<uint32_t, POS_DIMS>> min_max_cells_gpu(2);
        min_max_cells_gpu.memset(0xff, 1);
        min_max_cells_gpu.memset(0, 1, 1);
        tcnn::linear_kernel(compute_min_max_kernel<POS_DIMS>, 0, 0, points.n(), points.view(), cell_size, (uint32_t*)(min_max_cells_gpu.data()), (uint32_t*)(min_max_cells_gpu.data() + 1)); 
        Vector<uint32_t, POS_DIMS> min_max_cells[2];
        min_max_cells_gpu.copy_to_host(min_max_cells);
        min_point = min_max_cells[0].template cast<float>() * cell_size;
        max_point = min_max_cells[1].template cast<float>() * cell_size;
}

template<uint32_t POS_DIMS>
struct GridInfo {
    float cell_size;
    float cell_size_squared;
    Vectorxf<POS_DIMS> min;
    Vectorxf<POS_DIMS> max;
    Vectorxf<POS_DIMS> delta;
    Vector<uint32_t, POS_DIMS> dim;
    size_t num_cells;

    void update(const Vectorxf<POS_DIMS>& min_, const Vectorxf<POS_DIMS>& max_, uint32_t n_points, float cell_size_) {
        min = min_;
        max = max_;
        cell_size = cell_size_;
        cell_size_squared = cell_size * cell_size;
        Vectorxf<POS_DIMS> grid_size = max - min;
        dim = (grid_size.array() / cell_size).ceil().template cast<uint32_t>();
        dim.array() += 4; // add some padding to avoid boundary issues
        min.array() -= 2 * cell_size;
        max.array() += 2 * cell_size;
        grid_size = dim.template cast<float>() * cell_size;
        delta = dim.template cast<float>().cwiseQuotient(grid_size);
        num_cells = dim.prod();
    }

    void update(const tcnn::GPUMatrixDynamic<float>& positions, uint32_t n_points, float cell_size_) {
        Vectorxf<POS_DIMS> min_;
        Vectorxf<POS_DIMS> max_;
        compute_min_max<POS_DIMS>(positions, cell_size_, min_, max_);
        update(min_, max_, n_points, cell_size_);
    }
};

// calculate position in uniform grid
template<uint32_t POS_DIMS>
inline __device__ __host__ Vector<uint32_t, POS_DIMS> grid_pos(
    const Vectorxf<POS_DIMS> &p, 
    const Vectorxf<POS_DIMS> &world_origin, 
    float cell_size)
{
    return ((p - world_origin)/cell_size).template cast<uint32_t>(); 
}

template<uint32_t POS_DIMS>
inline __device__ __host__ uint32_t hash_grid_cell(const GridInfo<POS_DIMS>& grid, const Vector<uint32_t, POS_DIMS>& grid_pos) {
    uint32_t index = 0;
    uint32_t stride = 1;
    #pragma unroll
    for(int dim=0; dim< POS_DIMS; dim++) {
        index += grid_pos[dim]*stride;
        stride *= grid.dim[dim];
    }
    return index;
}

template<uint32_t POS_DIMS>
__global__ void calculate_hash(
    size_t num_particles,
    const GridInfo<POS_DIMS> grid_info,
    tcnn::MatrixView<float> position,
    uint32_t* __restrict__ hashes,
    uint32_t* __restrict__ particle_index)
{
    // https://github.com/NVIDIA/cuda-samples/blob/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_particles)
        return;
    
    particle_index[i] = i;

    Vectorxf<POS_DIMS> p;
    #pragma unroll
    for(int dim=0; dim<POS_DIMS; dim++) {
        p[dim] = position(dim, i);
    }

    Vector<uint32_t, POS_DIMS> grid_index = grid_pos<POS_DIMS>(p, grid_info.min, grid_info.cell_size);
    uint32_t hash = hash_grid_cell<POS_DIMS>(grid_info, grid_index);
    hashes[i] = hash;
}

template<typename T>
__global__ void find_cell_starts(
    uint32_t num_particles,
    const uint32_t* __restrict__ hashes,
    uint32_t* __restrict__ cell_start,
    uint32_t* __restrict__ cell_end
    )
{
    // Ref: https://github.com/NVIDIA/cuda-samples/blob/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint shared_hash[]; // blockSize + 1 elements
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t hash;
    if (i < num_particles) {
        hash = hashes[i];
        shared_hash[threadIdx.x + 1] = hash;
        if (i > 0 && threadIdx.x == 0) {
            shared_hash[0] = hashes[i - 1];
        }
    }
    cg::sync(cta);
    if (i < num_particles) {
        if (i == 0 || hash != shared_hash[threadIdx.x]) {
            cell_start[hash] = i; 
            if (i > 0)
                cell_end[shared_hash[threadIdx.x]] = i;
        }
        if (i == num_particles - 1) {
            cell_end[hash] = i + 1;
        }
    }
}

template<uint32_t POS_DIMS>
struct SearchStructure {
    uint32_t n_points = 0;
    float search_radius;
    GridInfo<POS_DIMS> grid_info;
    tcnn::GPUMemory<uint32_t> cell_start;
    tcnn::GPUMemory<uint32_t> cell_end;
    tcnn::GPUMemory<uint32_t> hash;
    tcnn::GPUMemory<uint32_t> sorted_to_unsorted_index;

    SearchStructure() {
    }

    SearchStructure& operator=(const SearchStructure& other) {
        n_points = other.n_points;
        search_radius = other.search_radius;
        grid_info = other.grid_info;
        cell_start = tcnn::GPUMemory<uint32_t>(other.cell_start); 
        cell_end = tcnn::GPUMemory<uint32_t>(other.cell_end);
        hash = tcnn::GPUMemory<uint32_t>(other.hash);
        sorted_to_unsorted_index = tcnn::GPUMemory<uint32_t>(other.sorted_to_unsorted_index);
        return *this;
    }

    void update(const tcnn::GPUMatrixDynamic<float>& positions, float search_radius_, const Vectorxf<POS_DIMS>& min_pos, const Vectorxf<POS_DIMS>& max_pos, cudaStream_t stream) {
        n_points = positions.n();
        search_radius = search_radius_;

        grid_info.update(min_pos, max_pos, n_points, search_radius);

        hash.resize(n_points);
        cell_start.resize(grid_info.num_cells); 
        cell_end.resize(grid_info.num_cells); 
        sorted_to_unsorted_index.resize(n_points);

        tcnn::linear_kernel(
            calculate_hash<POS_DIMS>, 
            0, stream,
            n_points, 
            grid_info, 
            positions.view(),
            hash.data(),
            sorted_to_unsorted_index.data());


        CUDA_CHECK_THROW(cudaMemsetAsync(cell_start.data(), 0xFFFFFFFF, grid_info.num_cells * sizeof(uint32_t), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(cell_end.data(), 0, grid_info.num_cells * sizeof(uint32_t), stream));

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            hash.data(),
            hash.data() + n_points,
            sorted_to_unsorted_index.data()); // cant be captured by cudaGraph :(
        
        uint32_t smemsize = sizeof(uint32_t) * (tcnn::n_threads_linear + 1);
        tcnn::linear_kernel(
            find_cell_starts<float>,
            smemsize, stream,
            n_points, 
            hash.data(), 
            cell_start.data(),
            cell_end.data()
        ); 
    }
};

PARTICLE_NAMESPACE_END 
