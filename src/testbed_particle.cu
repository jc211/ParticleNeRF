
#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/opengl_utils.h>

#include <particles/particle_encoding.h>

#ifdef NGP_GUI
#  include <imgui/imgui.h>
#  include <imgui/backends/imgui_impl_glfw.h>
#  include <imgui/backends/imgui_impl_opengl3.h>
#  include <imguizmo/ImGuizmo.h>
#  include <stb_image/stb_image.h>
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;
using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

particle::ParticleEncoding<precision_t>* Testbed::get_particle_encoding() const {
	if (auto* particle_encoding = dynamic_cast<particle::ParticleEncoding<precision_t>*>(m_encoding.get())) {
		return particle_encoding;
	} 
	return nullptr;
}

GPUMemory<float> Testbed::get_density_on_points(NerfPosition* positions, uint32_t n_points) {
	const uint32_t padded_n_points = tcnn::next_multiple(n_points, 128u); // n_points must be multiple of 128 for mixed precision inference
	// TODO(@jc211): this is a bit of a hack, but basically we are tresspassing in memory space to avoid a copy.
	GPUMemory<float> density(padded_n_points);
	const uint32_t batch_size = std::min(padded_n_points, 1u<<20);
	bool nerf_mode = m_testbed_mode == ETestbedMode::Nerf;

	const uint32_t padded_output_width = nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		network_precision_t
	>(m_stream.get(), &alloc, batch_size * padded_output_width);

	network_precision_t* mlp_out = std::get<0>(scratch);

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < padded_n_points; offset += batch_size) {
		uint32_t local_batch_size = std::min(padded_n_points - offset, batch_size);

		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, local_batch_size);

		GPUMatrix<float> positions_matrix((float*)(positions + offset), sizeof(NerfPosition)/sizeof(float), local_batch_size);
		if (nerf_mode) {
			m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
		} else {
			m_network->inference_mixed_precision(m_stream.get(), positions_matrix, density_matrix);
		}
		linear_kernel(tcnn::cast_from<network_precision_t>, 0, m_stream.get(), 
			local_batch_size, mlp_out, density.data() + offset);
	}
	return density;
}

#ifdef NGP_GUI
__device__ Eigen::Array3f paper_color_map(uint32_t x) {
	const Eigen::Array3f _col_data[3] = {
		{0.0f, 149.0f/255.0f, 1.0f},
		{213.0f/255.0f, 94.0f/255.0f, 0.0f},
		{1.0f, 204.0f/255.0f, 0.0f}
	};
	x = x % 3;
	return _col_data[x];
}

void Testbed::visualize_particle_encoding(const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center) {
	auto* pe = get_particle_encoding();
	if (!pe) {
		return;
	}

    static GLuint program = 0, vao = 0, vbos[2] = {0}, ind_vbo;
    static cudaGraphicsResource_t vbo_cuda[2], ind_vbo_cuda; 
    static uint32_t buffer_size = 0;
	static uint32_t level = 0xffffffff;
	bool refresh = false;
	if(level != m_n_particle_level) {
		level = m_n_particle_level;
		refresh = true;
	}
    auto num_particles = pe->n_particles(level); 
    if(!vao) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        for (int i = 0; i < 2/*std::size(vbos)*/; i++)
        {
            glGenBuffers(1, &vbos[i]);
            glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
            glBufferData(GL_ARRAY_BUFFER, num_particles * sizeof(Eigen::Vector3f), NULL, GL_DYNAMIC_DRAW);
            cudaGraphicsGLRegisterBuffer(&vbo_cuda[i], vbos[i], cudaGraphicsMapFlagsWriteDiscard);
        }
        glGenBuffers(1, &ind_vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ind_vbo);
        cudaGraphicsGLRegisterBuffer(&ind_vbo_cuda, ind_vbo, cudaGraphicsMapFlagsWriteDiscard);
    }

    if(buffer_size != num_particles) {
        for (int i = 0; i < 2 /*std::size(vbos)*/; i++)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
            glBufferData(GL_ARRAY_BUFFER, num_particles * sizeof(Eigen::Vector3f), NULL, GL_DYNAMIC_DRAW);
        }
    }

	static uint32_t update_id = 0;
    cudaStream_t stream = m_stream.get();
    if(update_id != pe->get_update_id(level) || refresh) { // starts at true
        Eigen::Vector3f* pos = (Eigen::Vector3f*)pe->get_particle_positions(level);
        uint32_t* ids = pe->get_particle_ids(level); 

		GPUMemoryArena::Allocation alloc;
		auto scratch = allocate_workspace_and_distribute<
			Eigen::Vector3f, Eigen::Vector3f           // alive 
		>(stream, &alloc, num_particles, num_particles); 

		Eigen::Vector3f* warp_pos = std::get<0>(scratch);
		Eigen::Vector3f* col = std::get<1>(scratch);

        // Multistream
        BoundingBox aabb = m_aabb;
		SyncedMultiStream streams{stream, 2};
        thrust::transform(thrust::cuda::par.on(streams.get(0)), pos, pos + num_particles, warp_pos, [aabb] __device__ (const Eigen::Vector3f& p) { 
            return aabb.min+p.cwiseProduct(aabb.diag());
		});
        thrust::transform(thrust::cuda::par.on(streams.get(1)), ids, ids + num_particles, (Eigen::Vector3f*)col, [num_particles] __device__ (uint32_t id) { 
			return paper_color_map(id);
		});

        void* ptr;
        size_t size;
        cudaGraphicsMapResources(2, vbo_cuda, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&ptr, &size, vbo_cuda[0]);
        if (ptr) CUDA_CHECK_THROW(cudaMemcpy(ptr, warp_pos, num_particles * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToDevice));
        cudaGraphicsResourceGetMappedPointer((void **)&ptr, &size, vbo_cuda[1]);
        if (ptr) CUDA_CHECK_THROW(cudaMemcpy(ptr, col, num_particles * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToDevice));
        cudaGraphicsUnmapResources(2, vbo_cuda, 0);
	    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }

    if (!program)
    {
        GLuint vs = compile_shader(false, R"foo(
            layout (location = 0) in vec3 pos;
            layout (location = 1) in vec3 col;
            out vec3 vtxcol;
            uniform vec2 f;
            uniform ivec2 res;
            uniform vec2 cen;
            uniform mat4 camera;
            void main()
            {
                vec4 p = camera * vec4(pos, 1.0);
                p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
                p.w = p.z;
                p.z = p.z - 0.1;
                p.xy += cen * p.w;
                vtxcol =  col;
                gl_Position = p; 
            }
            )foo");

        GLuint ps = compile_shader(true, R"foo(
            layout (location = 0) out vec4 o;
            in vec3 vtxcol;
            void main() {
                o = vec4(vtxcol, 0.5);
            }
            )foo");

        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, ps);
        glLinkProgram(program);
        if (!check_shader(program, "shader program", true))
        {
            glDeleteProgram(program);
            program = 0;
        }
    }


	Eigen::Matrix4f view2world=Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ind_vbo);
    glUseProgram(program);

	glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x()*2.f-1.f, screen_center.y()*-2.f+1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());

    GLuint posat = (GLuint)glGetAttribLocation(program, "pos");
    GLuint colat = (GLuint)glGetAttribLocation(program, "col");
    glEnableVertexAttribArray(posat);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glVertexAttribPointer(posat, 3, GL_FLOAT, GL_FALSE, 0 /* stride */, 0);

    glEnableVertexAttribArray(colat);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glVertexAttribPointer(colat, 3, GL_FLOAT, GL_FALSE, 0 /* stride */, 0);


    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0);
    // uint32_t offset = m_vis_particle_offset;
    // offset = std::min(offset, num_particles);
    // uint32_t requested_num = (uint32_t)(m_vis_particle_percent * num_particles / 100);
    // uint32_t count = std::min(requested_num, num_particles - offset);
    // glDrawArrays(GL_POINTS, offset /*first*/, count  /*count*/);
    glDrawArrays(GL_POINTS, 0 /*first*/, num_particles/*count*/);
    glUseProgram(0);
}
#endif
NGP_NAMESPACE_END