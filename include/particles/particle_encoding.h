#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/optimizer.h>

#include <particles/particles.h>

PARTICLE_NAMESPACE_BEGIN

template<typename T, uint32_t POS_DIMS, uint32_t FEATURE_DIMS>
__global__ void prune_kernel(
	uint32_t num_points,
	float threshold,

	const GridInfo<POS_DIMS> grid_info,
	const Vectorxf<POS_DIMS>* __restrict__ particles,
	const uint32_t* __restrict__ cell_start,
	const uint32_t* __restrict__ cell_end,
	const float* __restrict__ particle_feature,

    tcnn::MatrixView<const float> sigma,
    tcnn::MatrixView<uint32_t> alive
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;

    const Vectorxf<POS_DIMS> query = particles[query_index];

	float max_sigma = sigma(0, query_index);
	// float norm_features = 0;
	// for(uint32_t i = 0; i < FEATURE_DIMS; i++)
	// {
	// 	norm_features += particle_feature[i * num_points + query_index];
	// }
	// norm_features /= FEATURE_DIMS;

	int neighbours = 0;
    NEIGHBOUR_SEARCH({
        const Vectorxf<POS_DIMS>& neighbour = particles[neighbour_index];
        float dist = norm<POS_DIMS>(neighbour, query);
        if (dist < grid_info.cell_size) {
			max_sigma = max(max_sigma, sigma(0, neighbour_index));
			neighbours++;
        }
    });

	// if(max_sigma < threshold || norm_features < 0.01f) {
	if(max_sigma < threshold) {
		alive(0, query_index) = 0; // Deactivate point
	}
}

template<typename T, uint32_t N_FEATURE_DIMS, uint32_t POS_DIMS>
__global__ void forward_kernel(
	uint32_t num_points,
	RBFType rbf_type,

	const GridInfo<POS_DIMS> grid_info,
	const Vectorxf<POS_DIMS>* __restrict__ particles,
	const uint32_t* __restrict__ cell_start,
	const uint32_t* __restrict__ cell_end,

	const float* __restrict__ particle_feature,
	float* __restrict__ query_weights,
    tcnn::MatrixView<const float> data_in,
    tcnn::MatrixView<T> data_out
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;

    Vectorxf<POS_DIMS> query;
	#pragma unroll
	for(int dim = 0; dim < POS_DIMS; dim++) {
		query[dim] = data_in(dim, query_index);
	}
	uint32_t query_n_neighbors = 0;
	float search_radius = grid_info.cell_size*0.5f;
    float feature[N_FEATURE_DIMS] = {0.0f};
	// float total_weight = 0.0f;
    NEIGHBOUR_SEARCH({
        const Vectorxf<POS_DIMS>& neighbour = particles[neighbour_index];
        float dist = norm<POS_DIMS>(neighbour, query);
        if (dist < search_radius) {
            float weight = rbf(dist, search_radius, rbf_type);
			// total_weight += weight;
			query_n_neighbors++;
			const float* neighbour_feature = particle_feature + neighbour_index * N_FEATURE_DIMS;
			#pragma unroll
			for(int dim=0; dim<N_FEATURE_DIMS; dim++) {
				feature[dim] += weight*neighbour_feature[dim];
			}
		}
    });
	// query_weights[query_index] = total_weight;
	#pragma unroll
	for(int dim=0; dim<N_FEATURE_DIMS; dim++) {
		data_out(dim, query_index) = (T) (feature[dim]);
		// if(total_weight > 1e-8f) {
		// 	data_out(dim, query_index) = (T) (feature[dim]/total_weight);
		// }
	}
}


template<typename T, typename GRAD_T, uint32_t N_FEATURE_DIMS, uint32_t POS_DIMS>
__global__ void backward_kernel(
	uint32_t num_points,
	RBFType rbf_type,
	const GridInfo<POS_DIMS> grid_info,
	const Vectorxf<POS_DIMS>* __restrict__ positions,
	const uint32_t* __restrict__ cell_start,
	const uint32_t* __restrict__ cell_end,

	const float* __restrict__ particle_feature,
	const float* __restrict__ query_weights,

	const T* __restrict__ query_features,

	bool propagate_to_particle_positions,
	bool propagate_to_particle_features,
    tcnn::MatrixView<const float> positions_in,
    GRAD_T* __restrict__ pos_grad,
    GRAD_T* __restrict__ feature_grad,
    tcnn::MatrixView<const T> dL_dy,
    tcnn::MatrixView<float> dL_dinput
)
{
	uint query_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (query_index >= num_points) return;

    Vectorxf<POS_DIMS> query;
	#pragma unroll
	for(int dim=0; dim<POS_DIMS;dim++) {
		query[dim] = positions_in(dim, query_index);
	}
	
	//
	// float total_weight = query_weights[query_index];
	// if(total_weight < 1e-8f) return;
	// float inv_total_weight = 1.0f/total_weight;
	// const T* feature = query_features + query_index * N_FEATURE_DIMS;
	//

	bool compute_input_grads = dL_dinput ? true : false;
	Vectorxf<POS_DIMS> dL_dxq = Vectorxf<POS_DIMS>::Zero();
	float search_radius = grid_info.cell_size*0.5f;
    NEIGHBOUR_SEARCH({
        // neighbour, neighbour_index, squared_norm
        const Vectorxf<POS_DIMS>& neighbour = positions[neighbour_index];
		float dist = norm<POS_DIMS>(neighbour, query); 
		if (dist < search_radius && dist > 1e-6f) {
            GRAD_T* neighbour_pos_grad = pos_grad + neighbour_index*POS_DIMS;
            GRAD_T* neighbour_feature_grad = feature_grad + neighbour_index*N_FEATURE_DIMS;
			const float* neighbour_feature = particle_feature + neighbour_index*N_FEATURE_DIMS;
            float weight = rbf(dist, search_radius, rbf_type);
			float dL_dwi = 0;
			#pragma unroll
			for(int dim=0; dim<N_FEATURE_DIMS; dim++) {
				const float dLi_dy = (float)dL_dy(dim, query_index);
				//
				const float dL_dFi = weight* dLi_dy;
				dL_dwi += dLi_dy * neighbour_feature[dim];
				//

				//
				// const float dL_dFi = weight* dLi_dy*inv_total_weight;
				// dL_dwi += dLi_dy * (inv_total_weight*(neighbour_feature[dim] - ((float)(feature[dim]))));
				//

				if(propagate_to_particle_features) {
					atomicAdd(&(neighbour_feature_grad[dim]), (GRAD_T)dL_dFi); 
				}
			}
			if(propagate_to_particle_positions) {
				Vectorxf<POS_DIMS> dL_dxn =  dL_dwi * rbf_derivative(dist, search_radius, rbf_type) * norm_derivative<POS_DIMS>(neighbour, query);
				#pragma unroll
				for(int pos_dim = 0; pos_dim<POS_DIMS; pos_dim++) {
					atomicAdd(&(neighbour_pos_grad[pos_dim]), (GRAD_T)(dL_dxn[pos_dim]));
				}
			}
        }
    });
}

template<typename T>
struct ParticleLevelForwardContext : public tcnn::Context {
	tcnn::GPUMemory<float> weights;
};

template<typename T>
struct ParticleForwardContext : public tcnn::Context {
	std::vector<std::unique_ptr<tcnn::Context>> levels;
	tcnn::GPUMatrixDynamic<T>* density_network_output;
};

template <typename T>
class ParticleEncoding: public tcnn::Encoding<T> {
	public: 

	virtual void resize(cudaStream_t stream, uint32_t n_points, uint32_t level=0);
	virtual float get_search_radius(uint32_t level=0) const;
	virtual void set_search_radius(float radius, uint32_t level=0);

	virtual float* get_particle_positions(uint32_t level = 0) const;
	virtual uint32_t* get_particle_ids(uint32_t level = 0) const;
	virtual float* get_particle_features(uint32_t level=0) const;

	virtual uint32_t get_update_id(uint32_t level=0) const;
	virtual uint32_t get_n_levels() const;

	virtual uint32_t n_particles(uint32_t level=0);
	virtual void prune_particles(
		cudaStream_t stream,
		float threshold,
		const tcnn::GPUMatrixDynamic<float>& sigmas, uint32_t level);


    virtual void add_particles(cudaStream_t stream, uint32_t n_points, float* positions, float* features, uint32_t level=0);
    virtual void remove_particles(cudaStream_t stream, uint32_t* alive_mask, uint32_t level=0);
	virtual void physics_step(
		cudaStream_t stream,
		uint32_t n_steps,
		float nerf_scale,
		float constraint_softness,
		float min_distance,
		float velocity_damping,
		float timestep,
		bool use_collisions
	); 
	virtual void training_step(
		cudaStream_t stream, 
		bool use_physics);
	virtual ParticleInfo get_particle_info(uint32_t level=0) const;

    uint32_t m_dim_features;
	bool m_propagate_to_particle_positions = true;
	bool m_propagate_to_particle_features = true;
	RBFType m_rbf_type = RBFType::Bump;
};

template <typename T, uint32_t POS_DIMS = 3, uint32_t N_FEATURE_DIMS = 3>
class ParticleEncodingTemplated : public ParticleEncoding<T> {
	#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
		// The GPUs that we tested this on do not have an efficient 1D fp16
		// atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
		// forced to use 1D atomicAdds. As soon as 2D or higher is possible,
		// we can make use the efficient atomicAdd(half2) function.
		using grad_t = std::conditional_t<N_FEATURE_DIMS == 1, float, T>;
	#else
		// atomicAdd(__half2) is only supported with compute capability 60 and above.
		// Since atomicAdd(__half) is relatively slow / doesn't exist for low compute
		// capabilities, accumulate in fp32 instead.
		using grad_t = float;
	#endif
public:
	ParticleEncodingTemplated(const json& config)  {
		m_n_levels = config.value("n_levels", 1);
		m_n_padded_output_dims = m_n_output_dims = m_n_levels * N_FEATURE_DIMS;
		this->m_dim_features = N_FEATURE_DIMS;

		m_particles.resize(m_n_levels);
		m_search_radius.resize(m_n_levels);
		tcnn::default_rng_t rng;
		for(int level=0; level<m_n_levels; level++) {
			uint32_t n_particles = config["n_particles"][level];
			m_search_radius[level] = config["search_radius"][level];
			m_particles[level] = std::make_unique<Particles<T, POS_DIMS>>(n_particles, N_FEATURE_DIMS, config);
			m_particles[level]->init_random(rng, n_particles);
			m_particles[level]->update(0, m_search_radius[level]);
		}
	}

	std::unique_ptr<tcnn::Context> forward_impl_level(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input, 
		tcnn::GPUMatrixDynamic<T>* output = nullptr, 
		bool use_inference_params = false,
		bool prepare_input_gradients = false,
		uint32_t level = 0
	) {

        auto forward = std::make_unique<ParticleLevelForwardContext<T>>();
        const uint32_t num_elements = input.n();
		auto particles = m_particles[level];
		uint32_t n_particles = particles->size();
        if ((!output && !prepare_input_gradients) || m_n_padded_output_dims == 0 || num_elements == 0 || n_particles == 0) {
			return forward;
		}

		// forward->weights.resize(num_elements);
		const auto& search_struct = particles->m_particle_search_structure;
        tcnn::linear_kernel(forward_kernel<T, N_FEATURE_DIMS, POS_DIMS>,
            0, stream, num_elements, // num query points
			this->m_rbf_type,

			search_struct.grid_info,
			particles->get_position(),
			search_struct.cell_start.data(),
			search_struct.cell_end.data(),

            particles->get_features(),
			forward->weights.data(),
            input.view(), 
            output->view()
            );

        return forward;
    }

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream,
		const tcnn::GPUMatrixDynamic<float>& input, 
		tcnn::GPUMatrixDynamic<T>* output = nullptr, 
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
        auto forward = std::make_unique<ParticleForwardContext<T>>();
        const uint32_t num_elements = input.n();
        if ((!output && !prepare_input_gradients) || m_n_padded_output_dims == 0 || num_elements == 0) {
			return forward;
		}

		output->memset_async(stream, 0);
		forward->levels.resize(m_n_levels);
		tcnn::SyncedMultiStream streams(stream, m_n_levels);
		for(int i=0; i<m_n_levels; i++) {
			tcnn::GPUMatrixDynamic<T> level_output = output ? output->slice_rows(i*N_FEATURE_DIMS, N_FEATURE_DIMS) : tcnn::GPUMatrixDynamic<T>();
			auto level_context = forward_impl_level(streams.get(i), input, (output ? &level_output : nullptr), use_inference_params, prepare_input_gradients, i);
			forward->levels[i] = std::move(level_context);
		}
        return forward;
    }

    void backward_impl_level(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		const tcnn::GPUMatrixDynamic<T>* densities,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite,
		uint32_t level = 0
	) {

		const uint32_t num_elements = input.n();
		auto particles = m_particles[level];
		uint32_t n_particles = particles->size();

		if ((!dL_dinput && param_gradients_mode == tcnn::EGradientMode::Ignore) || m_n_padded_output_dims == 0 || num_elements == 0 || n_particles == 0) {
			return;
		}
		const auto& forward = dynamic_cast<const ParticleLevelForwardContext<T>&>(ctx);
		const auto& search_struct = particles->m_particle_search_structure;
		if (param_gradients_mode != tcnn::EGradientMode::Ignore) {
            tcnn::linear_kernel(backward_kernel<T, grad_t, N_FEATURE_DIMS, POS_DIMS>,
                0, stream, num_elements, // num query points
				this->m_rbf_type,
				search_struct.grid_info,
				particles->get_position(),
				search_struct.cell_start.data(),
				search_struct.cell_end.data(),

                particles->get_features(),
				forward.weights.data(),

				output.data(),
				this->m_propagate_to_particle_positions,
				this->m_propagate_to_particle_features,

                input.view(), // query points
				(T*)particles->get_position_gradient(),
				particles->get_feature_gradient(),
				dL_doutput.view(),

				dL_dinput != nullptr ? dL_dinput->view() : tcnn::MatrixView<float>()
            );
		}
    }

    void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const uint32_t num_elements = input.n();
		if ((!dL_dinput && param_gradients_mode == tcnn::EGradientMode::Ignore) || m_n_padded_output_dims == 0 || num_elements == 0) {
			return;
		}
		const auto& forward = dynamic_cast<const ParticleForwardContext<T>&>(ctx);
		tcnn::SyncedMultiStream streams(stream, m_n_levels);
		for(int level=0; level<m_n_levels; level++) {
			backward_impl_level(
				streams.get(level), 
				*(forward.levels[level]), 
				input, 
				output.slice_rows(level*N_FEATURE_DIMS, N_FEATURE_DIMS), 
				dL_doutput.slice_rows(level*N_FEATURE_DIMS, N_FEATURE_DIMS), 
				forward.density_network_output,  
				dL_dinput, use_inference_params, param_gradients_mode, level);
		}
    }

    void backward_backward_input_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<float>& dL_ddLdinput,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
        throw std::runtime_error("Not implemented");
    }

    uint32_t input_width() const override {
		return POS_DIMS;
	}

    uint32_t padded_output_width() const override {
		return m_n_output_dims + m_n_to_pad;
	}

	uint32_t output_width() const override {
		return m_n_output_dims;
	}

    uint32_t required_input_alignment() const override {
		return 1;
	}

    uint32_t required_output_alignment() const override {
		return this->m_dim_features * m_n_levels; 
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		CHECK_THROW(padded_output_width >= m_n_output_dims);
		m_n_to_pad = padded_output_width - m_n_output_dims;
	}

	tcnn::MatrixLayout preferred_output_layout() const override {
		return tcnn::MatrixLayout::AoS;
	}   
    

	size_t n_params() const override {
		return 16; // To trick nerf_network into calling the backwards
	}

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return {};
	}

	uint32_t dim_features() const {
		return this->m_dim_features;
	}

    json hyperparams() const override {
        throw std::runtime_error("Saving not implemented");
    }

	void resize(cudaStream_t stream, uint32_t n_points, uint32_t level) override {
		tcnn::default_rng_t rng;
		m_particles[level]->resize(n_points);
		m_particles[level]->init_random(rng, n_points, 0);
		m_particles[level]->update(stream, m_search_radius[level]);
	}

	float get_search_radius(uint32_t level) const override {
		return m_search_radius[level];
	}

	void set_search_radius(float radius, uint32_t level) override {
		m_search_radius[level] = radius;
		for(int i = 0; i < m_n_levels; i++) {
			m_particles[i]->update(0, radius);
		}
	}

	uint32_t n_particles(uint32_t level = 0) override {
		return m_particles[level]->size();
	}

	float* get_particle_positions(uint32_t level = 0) const override {
		return (float*)m_particles[level]->get_position(); 
	}

	uint32_t* get_particle_ids(uint32_t level = 0) const override {
		return m_particles[level]->get_ids(); 
	}

	float* get_particle_features(uint32_t level = 0) const override {
		return (float*)m_particles[level]->get_features(); 
	}

	uint32_t get_update_id(uint32_t level = 0) const override {
		return m_particles[level]->m_update_id;
	}
	uint32_t get_n_levels() const override {
		return m_n_levels; 
	}

    ParticleInfo get_particle_info(uint32_t level) const override { 
		return m_particles[level]->get_particle_info();
	};

    void add_particles(cudaStream_t stream, uint32_t n_points, float* positions, float* features, uint32_t level) override {
		m_particles[level]->add_particles(stream, n_points, (Vectorxf<POS_DIMS>*)positions, features);
	}

    void remove_particles(cudaStream_t stream, uint32_t* alive_mask, uint32_t level) override {
		m_particles[level]->remove_particles(stream, alive_mask);
	}

	void prune_particles(cudaStream_t stream, float threshold, const tcnn::GPUMatrixDynamic<float>& sigmas, uint32_t level) override {
		auto particles = m_particles[level];
		uint32_t num_particles = particles->size();
		if(num_particles == 0)
			return;

		if(sigmas.cols() != num_particles) {
			throw std::runtime_error("ParticleEncoding::prune_particles: sigmas.cols() != n_active_particles");
		}

		tcnn::GPUMemoryArena::Allocation alloc;
		auto scratch = tcnn::allocate_workspace_and_distribute<
			uint32_t           // alive 
		>(stream, &alloc, num_particles); 

		uint32_t* mask = std::get<0>(scratch);
		tcnn::GPUMatrix<uint32_t> alive_mask(mask, 1, num_particles);
		thrust::fill(thrust::cuda::par.on(stream), alive_mask.data(), alive_mask.data()+num_particles, 1);

		const auto& search_struct = particles->m_particle_search_structure;
		tcnn::linear_kernel(prune_kernel<T, POS_DIMS, N_FEATURE_DIMS>,
			0, stream, num_particles, 
			threshold,

			search_struct.grid_info,
			particles->get_position(),
			search_struct.cell_start.data(),
			search_struct.cell_end.data(),
			particles->get_features(),

			sigmas.view(),
			alive_mask.view()
			);
		particles->remove_particles(stream, mask);
	}

	void physics_step(
		cudaStream_t stream,
		uint32_t n_steps,
		float nerf_scale,
		float constraint_softness,
		float min_distance,
		float velocity_damping,
		float timestep,
		bool use_collisions
	) override {
		tcnn::SyncedMultiStream streams(stream, m_n_levels);
		for(int level=0; level<m_n_levels; level++) {
			m_particles[level]->physics_step(
				streams.get(level),
				n_steps,
				nerf_scale,
				constraint_softness,
				min_distance,
				velocity_damping, 
				timestep,
				use_collisions);
		}
	}

	void training_step(
		cudaStream_t stream, 
		bool use_physics) override {

		tcnn::SyncedMultiStream streams(stream, m_n_levels);
		for(int level=0; level<m_n_levels; level++) {
			auto particles = m_particles[level];	
			auto& particle_data = particles->get_particle_data();
			auto n_particles = particles->size();
			if(n_particles == 0)
				return;

			if(this->m_propagate_to_particle_features) {
				particle_data.feature_optimizer->step(streams.get(level), 1.0, particle_data.features.data(), nullptr, particle_data.features_gradient.data());
			}

			if(this->m_propagate_to_particle_positions) {
				tcnn::linear_kernel(clip_max_kernel<T, POS_DIMS>, 0, stream, n_particles, (T*)particle_data.positions_gradient.data(), 0.8f*m_search_radius[level]);
				if(!use_physics) {
					particle_data.position_optimizer->step(streams.get(level), 1.0, (float*)particle_data.positions.data(), nullptr, (T*)particle_data.positions_gradient.data());
				}
			}
			tcnn::linear_kernel(clip_max_kernel<T, POS_DIMS>, 0, stream, n_particles, (T*)particle_data.positions_gradient.data(), 0.8f*m_search_radius[level]);
			// cudaMemsetAsync((void*)particle_data.positions_gradient.data(), 0, n_particles*POS_DIMS*sizeof(T), streams.get(level));
			if(!use_physics && this->m_propagate_to_particle_positions) {
				cudaMemsetAsync((void*)particle_data.positions_gradient.data(), 0, n_particles*POS_DIMS*sizeof(T), streams.get(level));
				particles->remove_out_of_bounds(streams.get(level));
				// particles->update(streams.get(level), m_search_radius[level]);
			}
			cudaMemsetAsync((void*)particle_data.features_gradient.data(), 0, n_particles*this->m_dim_features*sizeof(T), streams.get(level));

		}
	}

    private:

    // derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;

	uint32_t m_n_levels;


    std::vector<std::shared_ptr<Particles<T, POS_DIMS>>> m_particles;
	std::vector<float> m_search_radius;
};

template <typename T>
ParticleEncoding<T>* create_particle_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	const uint32_t dim_features = encoding.value("dim_features", 2u);
	if(n_dims_to_encode == 3) {
		switch (dim_features) {
			case 2: return new ParticleEncodingTemplated<T, 3u, 2u>{encoding};
			case 4: return new ParticleEncodingTemplated<T, 3u, 4u>{encoding};
			case 8: return new ParticleEncodingTemplated<T, 3u, 8u>{encoding};
			case 16: return new ParticleEncodingTemplated<T, 3u, 16u>{encoding};
			case 32: return new ParticleEncodingTemplated<T, 3u, 32u>{encoding};
			case 64: return new ParticleEncodingTemplated<T, 3u, 64u>{encoding};
			default: throw std::runtime_error{"Particle Encoding: dim_features must be 2, 4, 8, 16, 32, or 64."};
		}
	} 
	else if(n_dims_to_encode == 2) {
		switch (dim_features) {
			case 2: return new ParticleEncodingTemplated<T, 2u, 2u>{encoding};
			case 4: return new ParticleEncodingTemplated<T, 2u, 4u>{encoding};
			case 8: return new ParticleEncodingTemplated<T, 2u, 8u>{encoding};
			case 16: return new ParticleEncodingTemplated<T, 2u, 16u>{encoding};
			case 32: return new ParticleEncodingTemplated<T, 2u, 32u>{encoding};
			case 64: return new ParticleEncodingTemplated<T, 2u, 64u>{encoding};
			default: throw std::runtime_error{"Particle Encoding: dim_features must be 2, 4, 8, 16, 32, or 64."};
		}
	}
	throw std::runtime_error{"Particle Encoding can only encode 2 or 3 dimensions"};
}

PARTICLE_NAMESPACE_END