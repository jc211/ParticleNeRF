
#include <neural-graphics-primitives/opengl_utils.h>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

#include <vector>

NGP_NAMESPACE_BEGIN
#ifdef NGP_GUI
void glCheckError(const char* file, unsigned int line) {
	GLenum errorCode = glGetError();
	while (errorCode != GL_NO_ERROR) {
		std::string fileString(file);
		std::string error = "unknown error";
		// clang-format off
		switch (errorCode) {
			case GL_INVALID_ENUM:      error = "GL_INVALID_ENUM"; break;
			case GL_INVALID_VALUE:     error = "GL_INVALID_VALUE"; break;
			case GL_INVALID_OPERATION: error = "GL_INVALID_OPERATION"; break;
			case GL_STACK_OVERFLOW:    error = "GL_STACK_OVERFLOW"; break;
			case GL_STACK_UNDERFLOW:   error = "GL_STACK_UNDERFLOW"; break;
			case GL_OUT_OF_MEMORY:     error = "GL_OUT_OF_MEMORY"; break;
		}
		// clang-format on

		tlog::error() << "OpenglError : file=" << file << " line=" << line << " error:" << error;
		errorCode = glGetError();
	}
}

bool check_shader(uint32_t handle, const char* desc, bool program) {
	GLint status = 0, log_length = 0;
	if (program) {
		glGetProgramiv(handle, GL_LINK_STATUS, &status);
		glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &log_length);
	} else {
		glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &log_length);
	}
	if ((GLboolean)status == GL_FALSE) {
		tlog::error() << "Failed to compile shader: " << desc;
	}
	if (log_length > 1) {
		std::vector<char> log; log.resize(log_length+1);
		if (program) {
			glGetProgramInfoLog(handle, log_length, NULL, (GLchar*)log.data());
		} else {
			glGetShaderInfoLog(handle, log_length, NULL, (GLchar*)log.data());
		}
		log.back() = 0;
		tlog::error() << log.data();
	}
	return (GLboolean)status == GL_TRUE;
}

uint32_t compile_shader(bool pixel, const char* code) {
	GLuint g_VertHandle = glCreateShader(pixel ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER );
	const char* glsl_version = "#version 330\n";
	const GLchar* strings[2] = { glsl_version, code};
	glShaderSource(g_VertHandle, 2, strings, NULL);
	glCompileShader(g_VertHandle);
	if (!check_shader(g_VertHandle, pixel?"pixel":"vertex", false)) {
		glDeleteShader(g_VertHandle);
		return 0;
	}
	return g_VertHandle;
}

#endif //NGP_GUI
NGP_NAMESPACE_END