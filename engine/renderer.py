from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    glUseProgram, glUniformMatrix4fv, glDrawArrays, glEnable,
    glGetUniformLocation, glUniform3f,glUniform1f,
    glActiveTexture, glBindTexture, glUniform1i,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    GL_LINK_STATUS, GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_DEPTH_TEST,
    GL_TEXTURE_2D, GL_TEXTURE0
)
import numpy as np


# =========================
# Shader Quellen
# =========================

GRID_VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;

out vec3 v_world_pos;

void main()
{
    vec4 world = u_model * vec4(aPos, 1.0);
    v_world_pos = world.xyz;
    gl_Position = u_proj * u_view * world;
}
"""

GRID_FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 v_world_pos;
out vec4 FragColor;

uniform vec3 u_color;

void main()
{
    // simple grid-style coloring
    float scale = 1.0;
    float line = step(0.999, abs(sin(v_world_pos.x * scale))) +
                 step(0.999, abs(sin(v_world_pos.z * scale)));

    vec3 base = vec3(0.15, 0.15, 0.17);

    // white grid lines
    vec3 grid_color = vec3(1.0, 1.0, 1.0);

    float alpha = clamp(line, 0.0, 1.0) * 0.25; // <<< grid transparency

    vec3 color = mix(base, grid_color, clamp(line, 0.0, 1.0));
    FragColor = vec4(color, alpha);
}
"""

OBJECT_VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 vWorldPos;
out vec3 vNormal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    vec4 world = u_model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;

    // transform normal to world space
    mat3 normalMatrix = mat3(transpose(inverse(u_model)));
    vNormal = normalize(normalMatrix * aNormal);

    gl_Position = u_proj * u_view * world;
}
"""

OBJECT_FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

out vec4 FragColor;

uniform vec3 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform float u_triplanar_scale;

void main()
{
    if (!u_use_texture)
    {
        FragColor = vec4(u_color, 1.0);
        return;
    }

    vec3 n = normalize(vNormal);
    vec3 blend = abs(n);
    blend /= (blend.x + blend.y + blend.z);

    vec3 signN = sign(vNormal);

    vec2 uvX = vec2(vWorldPos.z * signN.x, vWorldPos.y) * u_triplanar_scale;
    vec2 uvY = vec2(vWorldPos.x, vWorldPos.z * signN.y) * u_triplanar_scale;
    vec2 uvZ = vec2(vWorldPos.x * signN.z, vWorldPos.y) * u_triplanar_scale;

    vec4 tx = texture(u_texture, uvX);
    vec4 ty = texture(u_texture, uvY);
    vec4 tz = texture(u_texture, uvZ);

    FragColor = tx * blend.x + ty * blend.y + tz * blend.z;
}
"""


# =========================
# Shader Utils
# =========================

def compile_shader(src, shader_type):
    """
    Docstring für compile_shader
    
    :param src: The shader source code
    :param shader_type: The type of shader (vertex or fragment)
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader)
        error = error.decode() if error else "<no shader log>"
        raise RuntimeError(error)

    return shader


def create_program(vs_src, fs_src):
    """
    Docstring für create_program

    :param vs_src: The vertex shader source
    :param fs_src: The fragment shader source
    """
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program)
        error = error.decode() if error else "<no program log>"
        raise RuntimeError(error)

    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


# =========================
# Renderer
# =========================

class Renderer:
    def __init__(self, plane_size=50.0):
        """
        Docstring für __init__

        :param self: The object itself
        :param plane_size: The size of the ground plane
        """
        self.grid_program   = create_program(GRID_VERTEX_SHADER_SRC, GRID_FRAGMENT_SHADER_SRC)
        self.object_program = create_program(OBJECT_VERTEX_SHADER_SRC, OBJECT_FRAGMENT_SHADER_SRC)

        # static ground plane
        self._create_plane(plane_size)

        glEnable(GL_DEPTH_TEST)

        # grid uniforms
        self.grid_u_view  = glGetUniformLocation(self.grid_program, "u_view")
        self.grid_u_proj  = glGetUniformLocation(self.grid_program, "u_proj")
        self.grid_u_model = glGetUniformLocation(self.grid_program, "u_model")

        # object uniforms
        self.obj_u_view  = glGetUniformLocation(self.object_program, "u_view")
        self.obj_u_proj  = glGetUniformLocation(self.object_program, "u_proj")
        self.obj_u_model = glGetUniformLocation(self.object_program, "u_model")
        self.obj_u_color = glGetUniformLocation(self.object_program, "u_color")

        self.obj_u_texture     = glGetUniformLocation(self.object_program, "u_texture")
        self.obj_u_use_texture = glGetUniformLocation(self.object_program, "u_use_texture")

        self.obj_u_triplanar_scale = glGetUniformLocation(
            self.object_program, "u_triplanar_scale"
        )

        self.model = np.identity(4, dtype=np.float32)

    def _create_plane(self, size):
        """
        Docstring für _create_plane

        :param self: The object itself
        :param size: The size of the plane
        """
        # XZ plane at Y = 0
        vertices = np.array([
            -size, 0.0, -size,
             size, 0.0, -size,
             size, 0.0,  size,

            -size, 0.0, -size,
             size, 0.0,  size,
            -size, 0.0,  size,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        self.vertex_count = 6

    def draw_plane(self, camera, aspect):
        """
        Docstring für draw_plane

        :param self: The object itself
        :param camera: The camera object
        :param aspect: The aspect ratio of the viewport
        """
        glUseProgram(self.grid_program)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)

        glUniformMatrix4fv(self.grid_u_view, 1, GL_TRUE, view)
        glUniformMatrix4fv(self.grid_u_proj, 1, GL_TRUE, proj)
        glUniformMatrix4fv(self.grid_u_model, 1, GL_TRUE, self.model)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def draw_object(self, obj, camera, aspect):
        """
        Docstring für draw_object

        :param self: The object itself
        :param obj: The object to draw
        :param camera: The camera object
        :param aspect: The aspect ratio of the viewport
        """
        glUseProgram(self.object_program)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)
        model = obj.transform.matrix()

        glUniformMatrix4fv(self.obj_u_view, 1, GL_TRUE, view)
        glUniformMatrix4fv(self.obj_u_proj, 1, GL_TRUE, proj)
        glUniformMatrix4fv(self.obj_u_model, 1, GL_TRUE, model)

        # material color fallback
        glUniform3f(self.obj_u_color, *obj.material.color)

        # texture binding (if present)
        if obj.material and obj.material.texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, obj.material.texture)
            glUniform1i(self.obj_u_texture, 0)
            glUniform1i(self.obj_u_use_texture, 1)
        else:
            glUniform1i(self.obj_u_use_texture, 0)

        # triplanar scale for texture mapping on objects
        glUniform1f(self.obj_u_triplanar_scale, 0.2)

        obj.mesh.draw()