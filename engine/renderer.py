from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
    glUseProgram, glUniformMatrix4fv, glDrawArrays, glEnable,
    glGetUniformLocation, glUniform3f, glUniform1f,glTexSubImage2D,
    glActiveTexture, glBindTexture, glUniform1i,glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glUniform3fv, glUniform2f,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    GL_LINK_STATUS, GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_DEPTH_TEST,
    GL_TEXTURE_2D, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_RGB,
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
    GL_FRAMEBUFFER_COMPLETE,
    GL_NEAREST, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_DEPTH_COMPONENT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_RED
)
import numpy as np
import ctypes


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

// lighting
uniform vec3 u_light_pos;        // point light (lamp)
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_ambient_strength;

// SSAO
uniform sampler2D u_ssao;
uniform vec2 u_screen_size;
uniform bool u_emissive;

void main()
{
    vec3 baseColor;

    // ---------- texture / color ----------
    if (u_use_texture)
    {
        vec3 n = normalize(vNormal);
        vec3 blend = abs(n);
        blend /= (blend.x + blend.y + blend.z);

        vec3 signN = sign(n);

        vec2 uvX = vec2(vWorldPos.z * signN.x, vWorldPos.y) * u_triplanar_scale;
        vec2 uvY = vec2(vWorldPos.x, vWorldPos.z * signN.y) * u_triplanar_scale;
        vec2 uvZ = vec2(vWorldPos.x * signN.z, vWorldPos.y) * u_triplanar_scale;

        vec4 tx = texture(u_texture, uvX);
        vec4 ty = texture(u_texture, uvY);
        vec4 tz = texture(u_texture, uvZ);

        baseColor = (tx * blend.x + ty * blend.y + tz * blend.z).rgb;
    }
    else
    {
        baseColor = u_color;
    }

    // ---------- emissive (light source itself) ----------
    if (u_emissive)
    {
        FragColor = vec4(baseColor, 1.0);
        return;
    }

    // ---------- lighting ----------
    vec3 n = normalize(vNormal);

    // ambient
    vec3 ambient = u_ambient_strength * baseColor;

    // point light (lamp)
    vec3 lightDir = u_light_pos - vWorldPos;
    float distance = length(lightDir);
    lightDir = normalize(lightDir);

    float diff = max(dot(n, lightDir), 0.0);

    // inverse-square attenuation (stable)
    float attenuation = u_light_intensity / (distance * distance + 1.0);

    vec3 diffuse = diff * baseColor * u_light_color * attenuation;

    // SSAO sampling
    vec2 uv = gl_FragCoord.xy / u_screen_size;
    float ao = clamp(texture(u_ssao, uv).r, 0.25, 1.0);

    vec3 finalColor = (ambient + diffuse) * ao;
    FragColor = vec4(finalColor, 1.0);
}
"""

NORMAL_FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 vNormal;
out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal) * 0.5 + 0.5;
    FragColor = vec4(n, 1.0);
}
"""

SSAO_VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;

void main()
{
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

SSAO_FRAGMENT_SHADER_SRC = """
#version 330 core
out float FragColor;
in vec2 vUV;

uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform sampler2D u_noise;

uniform vec3 u_samples[64];
uniform mat4 u_proj;

uniform vec2 u_noise_scale;
uniform float u_radius;
uniform float u_bias;

void main()
{
    vec3 normal = normalize(texture(u_normal, vUV).xyz * 2.0 - 1.0);
    float depth = texture(u_depth, vUV).r;

    vec3 randomVec = texture(u_noise, vUV * u_noise_scale).xyz;
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;

    for (int i = 0; i < 64; i++)
    {
        vec3 samplePos = TBN * u_samples[i];
        samplePos = samplePos * u_radius + vec3(0.0, 0.0, depth);

        vec4 offset = vec4(samplePos, 1.0);
        offset = u_proj * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(u_depth, offset.xy).r;
        float rangeCheck = smoothstep(0.0, 1.0, u_radius / abs(depth - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + u_bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / 64.0);
    FragColor = occlusion;
}
"""

SSAO_BLUR_FRAGMENT_SHADER_SRC = """
#version 330 core
out float FragColor;
in vec2 vUV;
uniform sampler2D u_ssao;

void main()
{
    float result = 0.0;
    vec2 texel = 1.0 / textureSize(u_ssao, 0);

    for (int x = -2; x <= 2; x++)
    for (int y = -2; y <= 2; y++)
        result += texture(u_ssao, vUV + vec2(x, y) * texel).r;

    FragColor = result / 25.0;
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

        self.normal_program = create_program(
            OBJECT_VERTEX_SHADER_SRC,
            NORMAL_FRAGMENT_SHADER_SRC
        )

        # SSAO programs
        self.ssao_program = create_program(SSAO_VERTEX_SHADER_SRC, SSAO_FRAGMENT_SHADER_SRC)
        self.ssao_blur_program = create_program(SSAO_VERTEX_SHADER_SRC, SSAO_BLUR_FRAGMENT_SHADER_SRC)

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
        self.obj_u_emissive = glGetUniformLocation(
            self.object_program, "u_emissive"
        )

        # lighting uniforms
        self.obj_u_light_pos = glGetUniformLocation(self.object_program, "u_light_pos")
        self.obj_u_light_color = glGetUniformLocation(self.object_program, "u_light_color")
        self.obj_u_light_intensity = glGetUniformLocation(
            self.object_program, "u_light_intensity"
        )
        self.obj_u_ambient_strength = glGetUniformLocation(
            self.object_program, "u_ambient_strength"
        )

        # SSAO uniforms for object shader
        self.obj_u_ssao = glGetUniformLocation(self.object_program, "u_ssao")
        self.obj_u_screen_size = glGetUniformLocation(
            self.object_program, "u_screen_size"
        )

        # ---------- SSAO G-buffer ----------
        self.gbuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.gbuffer)

        # normal texture
        self.g_normal = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.g_normal)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            1280, 720, 0, GL_RGBA, GL_UNSIGNED_BYTE, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.g_normal, 0
        )

        # depth texture
        self.g_depth = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.g_depth)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            1280, 720, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_2D, self.g_depth, 0
        )

        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # SSAO kernel
        self.ssao_kernel = []
        for i in range(64):
            sample = np.random.uniform(-1.0, 1.0, 3)
            sample[2] = np.random.uniform(0.0, 1.0)
            sample = sample / np.linalg.norm(sample)
            scale = i / 64.0
            scale = 0.1 + 0.9 * scale * scale
            self.ssao_kernel.append(sample * scale)

        self.ssao_kernel = np.array(self.ssao_kernel, dtype=np.float32)

        # SSAO noise
        noise = np.random.uniform(-1.0, 1.0, (16, 3)).astype(np.float32)
        self.ssao_noise_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.ssao_noise_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_FLOAT, noise)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # SSAO framebuffer and texture
        self.ssao_fbo = glGenFramebuffers(1)
        self.ssao_tex = glGenTextures(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.ssao_fbo)
        glBindTexture(GL_TEXTURE_2D, self.ssao_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1280, 720, 0, GL_RED, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.ssao_tex, 0)
        # initialize SSAO texture to white (AO = 1.0 default)
        white = np.ones((720, 1280), dtype=np.float32)
        glBindTexture(GL_TEXTURE_2D, self.ssao_tex)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            1280, 720,
            GL_RED, GL_FLOAT, white
        )
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # fullscreen quad
        quad = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1, -1,  0, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ], dtype=np.float32)

        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))

        glBindVertexArray(0)

        self.model = np.identity(4, dtype=np.float32)

        # active light (single-light pipeline)
        self.light_pos = (0.0, 0.0, 0.0)
        self.light_color = (1.0, 1.0, 1.0)
        self.light_intensity = 0.0
    def set_light(self, position, color, intensity):
        """
        Set the active point light (used by all objects).
        """
        self.light_pos = position
        self.light_color = color
        self.light_intensity = intensity

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

        # emissive flag (sun / lamps)
        is_emissive = obj.material is not None and getattr(obj.material, "emissive", False)
        glUniform1i(self.obj_u_emissive, int(is_emissive))

        # SSAO texture
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.ssao_tex)
        glUniform1i(self.obj_u_ssao, 1)
        glUniform2f(self.obj_u_screen_size, 1280.0, 720.0)

        # triplanar scale for texture mapping on objects
        glUniform1f(self.obj_u_triplanar_scale, 0.1)

        # lighting (dynamic light from world)
        glUniform3f(self.obj_u_light_pos, *self.light_pos)
        glUniform3f(self.obj_u_light_color, *self.light_color)
        glUniform1f(self.obj_u_light_intensity, self.light_intensity)
        glUniform1f(self.obj_u_ambient_strength, 0.25)

        obj.mesh.draw()

    def render_normals(self, objects, camera, aspect):
        glBindFramebuffer(GL_FRAMEBUFFER, self.gbuffer)
        glUseProgram(self.normal_program)
        glEnable(GL_DEPTH_TEST)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)

        for obj in objects:
            model = obj.transform.matrix()

            glUniformMatrix4fv(
                glGetUniformLocation(self.normal_program, "u_view"),
                1, GL_TRUE, view
            )
            glUniformMatrix4fv(
                glGetUniformLocation(self.normal_program, "u_proj"),
                1, GL_TRUE, proj
            )
            glUniformMatrix4fv(
                glGetUniformLocation(self.normal_program, "u_model"),
                1, GL_TRUE, model
            )

            obj.mesh.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_ssao(self, camera, width, height):
        glBindFramebuffer(GL_FRAMEBUFFER, self.ssao_fbo)
        glUseProgram(self.ssao_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.g_normal)
        glUniform1i(glGetUniformLocation(self.ssao_program, "u_normal"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.g_depth)
        glUniform1i(glGetUniformLocation(self.ssao_program, "u_depth"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.ssao_noise_tex)
        glUniform1i(glGetUniformLocation(self.ssao_program, "u_noise"), 2)

        for i in range(64):
            glUniform3fv(
                glGetUniformLocation(self.ssao_program, f"u_samples[{i}]"),
                1, self.ssao_kernel[i]
            )

        glUniformMatrix4fv(
            glGetUniformLocation(self.ssao_program, "u_proj"),
            1, GL_TRUE, camera.get_projection_matrix(width / height)
        )

        glUniform2f(
            glGetUniformLocation(self.ssao_program, "u_noise_scale"),
            width / 4.0, height / 4.0
        )

        glUniform1f(glGetUniformLocation(self.ssao_program, "u_radius"), 0.25)
        glUniform1f(glGetUniformLocation(self.ssao_program, "u_bias"), 0.05)

        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)