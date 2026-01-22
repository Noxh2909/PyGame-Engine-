import ctypes
import math
import random
from typing import Optional

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import numpy as np
from OpenGL import GL

# =========================
# Shader Utils
# =========================


def load_shader(path: str) -> str:
    """
    Docstring für load_shader
    
    :param path: Path to the shader file
    :type path: str
    :return: The source code of the shader as a string
    :rtype: str
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compile_shader(source: str, shader_type: int) -> int:
    """Compile a GLSL shader from source and return the handle.

    :param source: The shader source code as a single string.
    :param shader_type: GL.GL_VERTEX_SHADER, GL.GL_FRAGMENT_SHADER or GL.GL_GEOMETRY_SHADER.
    :raises RuntimeError: On compilation failure.
    """
    shader = GL.glCreateShader(shader_type)
    if shader is None or shader == 0:
        raise RuntimeError("Failed to create shader")
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    # Check compile status and raise an error if compilation failed
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        info = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {info}")
    return shader


def link_program(vertex_src: str, fragment_src: str, geometry_src: Optional[str] = None) -> int:
    """Link a GLSL program from supplied shader sources.

    :param vertex_src: Vertex shader source code.
    :param fragment_src: Fragment shader source code.
    :param geometry_src: Optional geometry shader source code.
    :return: OpenGL program handle.
    :raises RuntimeError: On linking failure.
    """
    program = GL.glCreateProgram()
    if program is None or program == 0:
        raise RuntimeError("Failed to create program")
    vs = compile_shader(vertex_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    gs = None
    if geometry_src:
        gs = compile_shader(geometry_src, GL.GL_GEOMETRY_SHADER)
        GL.glAttachShader(program, gs)
    GL.glLinkProgram(program)
    # Check link status
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        info = GL.glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {info}")
    # Shaders can be deleted once linked
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    if gs:
        GL.glDeleteShader(gs)
    return program


# =========================
# Shader Quellen
# =========================

GRID_VERTEX_SHADER_SRC = load_shader("engine/rendering/shader/grid.vert")
GRID_FRAGMENT_SHADER_SRC = load_shader("engine/rendering/shader/grid.frag")
OBJECT_VERTEX_SHADER_SRC = load_shader("engine/rendering/shader/object.vert")
OBJECT_FRAGMENT_SHADER_SRC = load_shader("engine/rendering/shader/object.frag")
NORMAL_FRAGMENT_SHADER_SRC = load_shader("engine/rendering/shader/normal.frag")
SSAO_VERTEX_SHADER_SRC = load_shader("engine/rendering/shader/ssao.vert")
SSAO_FRAGMENT_SHADER_SRC = load_shader("engine/rendering/shader/ssao.frag")
SSAO_BLUR_FRAGMENT_SHADER_SRC = load_shader("engine/rendering/shader/ssao_blur.frag")

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
        self.grid_program = link_program(
            GRID_VERTEX_SHADER_SRC, GRID_FRAGMENT_SHADER_SRC
        )
        self.object_program = link_program(
            OBJECT_VERTEX_SHADER_SRC, OBJECT_FRAGMENT_SHADER_SRC
        )

        self.normal_program = link_program(
            OBJECT_VERTEX_SHADER_SRC, NORMAL_FRAGMENT_SHADER_SRC
        )

        # SSAO programs
        self.ssao_program = link_program(
            SSAO_VERTEX_SHADER_SRC, SSAO_FRAGMENT_SHADER_SRC
        )
        self.ssao_blur_program = link_program(
            SSAO_VERTEX_SHADER_SRC, SSAO_BLUR_FRAGMENT_SHADER_SRC
        )

        # static ground plane
        self._create_plane(plane_size)

        GL.glEnable(GL.GL_DEPTH_TEST)

        # grid uniforms
        self.grid_u_view = GL.glGetUniformLocation(self.grid_program, "u_view")
        self.grid_u_proj = GL.glGetUniformLocation(self.grid_program, "u_proj")
        self.grid_u_model = GL.glGetUniformLocation(self.grid_program, "u_model")

        # object uniforms
        self.obj_u_view = GL.glGetUniformLocation(self.object_program, "u_view")
        self.obj_u_proj = GL.glGetUniformLocation(self.object_program, "u_proj")
        self.obj_u_model = GL.glGetUniformLocation(self.object_program, "u_model")
        self.obj_u_color = GL.glGetUniformLocation(self.object_program, "u_color")

        self.obj_u_texture = GL.glGetUniformLocation(self.object_program, "u_texture")
        self.obj_u_use_texture = GL.glGetUniformLocation(
            self.object_program, "u_use_texture"
        )

        self.obj_u_texture_mode = GL.glGetUniformLocation(
            self.object_program, "u_texture_mode"
        )

        self.obj_u_triplanar_scale = GL.glGetUniformLocation(
            self.object_program, "u_triplanar_scale"
        )

        self.obj_u_emissive = GL.glGetUniformLocation(self.object_program, "u_emissive")

        # lighting uniforms
        self.obj_u_light_pos = GL.glGetUniformLocation(self.object_program, "u_light_pos")
        self.obj_u_light_color = GL.glGetUniformLocation(
            self.object_program, "u_light_color"
        )
        self.obj_u_light_intensity = GL.glGetUniformLocation(
            self.object_program, "u_light_intensity"
        )
        self.obj_u_ambient_strength = GL.glGetUniformLocation(
            self.object_program, "u_ambient_strength"
        )

        # SSAO uniforms for object shader
        self.obj_u_ssao = GL.glGetUniformLocation(self.object_program, "u_ssao")
        self.obj_u_screen_size = GL.glGetUniformLocation(
            self.object_program, "u_screen_size"
        )

        # ---------- SSAO G-buffer ----------
        self.gbuffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.gbuffer)

        # normal texture
        self.g_normal = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_normal)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 1280, 720, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.g_normal, 0
        )

        # depth texture
        self.g_depth = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_depth)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT,
            1280,
            720,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.g_depth, 0
        )

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

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
        self.ssao_noise_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_noise_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 4, 4, 0, GL.GL_RGB, GL.GL_FLOAT, noise)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

        # SSAO framebuffer and texture
        self.ssao_fbo = GL.glGenFramebuffers(1)
        self.ssao_tex = GL.glGenTextures(1)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_fbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RED, 1280, 720, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.ssao_tex, 0
        )
        # initialize SSAO texture to white (AO = 1.0 default)
        white = np.ones((720, 1280), dtype=np.float32)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 1280, 720, GL.GL_RED, GL.GL_FLOAT, white)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # fullscreen quad
        quad = np.array(
            [
                -1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                0,
                1,
                1,
                1,
                1,
                -1,
                -1,
                0,
                0,
                1,
                1,
                1,
                1,
                -1,
                1,
                0,
                1,
            ],
            dtype=np.float32,
        )

        self.quad_vao = GL.glGenVertexArrays(1)
        self.quad_vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)

        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, ctypes.c_void_p(8))

        GL.glBindVertexArray(0)

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
        vertices = np.array(
            [
                -size,
                0.0,
                -size,
                size,
                0.0,
                -size,
                size,
                0.0,
                size,
                -size,
                0.0,
                -size,
                size,
                0.0,
                size,
                -size,
                0.0,
                size,
            ],
            dtype=np.float32,
        )

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindVertexArray(0)

        self.vertex_count = 6

    def draw_plane(self, camera, aspect):
        """
        Docstring für draw_plane

        :param self: The object itself
        :param camera: The camera object
        :param aspect: The aspect ratio of the viewport
        """
        GL.glUseProgram(self.grid_program)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)

        GL.glUniformMatrix4fv(self.grid_u_view, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(self.grid_u_proj, 1, GL.GL_TRUE, proj)
        GL.glUniformMatrix4fv(self.grid_u_model, 1, GL.GL_TRUE, self.model)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)

    def draw_object(self, obj, camera, aspect):
        """
        Docstring für draw_object

        :param self: The object itself
        :param obj: The object to draw
        :param camera: The camera object
        :param aspect: The aspect ratio of the viewport
        """
        GL.glUseProgram(self.object_program)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)
        model = obj.transform.matrix()

        GL.glUniformMatrix4fv(self.obj_u_view, 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(self.obj_u_proj, 1, GL.GL_TRUE, proj)
        GL.glUniformMatrix4fv(self.obj_u_model, 1, GL.GL_TRUE, model)

        # material color fallback
        GL.glUniform3f(self.obj_u_color, *obj.material.color)

        # texture binding (if present)
        if obj.material and obj.material.texture is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, obj.material.texture)
            GL.glUniform1i(self.obj_u_texture, 0)
            GL.glUniform1i(self.obj_u_use_texture, 1)
        else:
            GL.glUniform1i(self.obj_u_use_texture, 0)

        # emissive flag (sun / lamps)
        is_emissive = obj.material is not None and getattr(
            obj.material, "emissive", False
        )
        GL.glUniform1i(self.obj_u_emissive, int(is_emissive))

        # SSAO texture
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_tex)
        GL.glUniform1i(self.obj_u_ssao, 1)
        GL.glUniform2f(self.obj_u_screen_size, 1280.0, 720.0)

        mat = obj.material
        mode = getattr(mat, "texture_scale_mode", None) or "default"

        # -------- default: UV mapping (object-local) --------
        if mode == "default":
            # Shader: u_texture_mode == 0 → UV-Mapping
            GL.glUniform1i(self.obj_u_texture_mode, 0)

            # triplanar scale wird hier NICHT benutzt
            GL.glUniform1f(self.obj_u_triplanar_scale, 1.0)

        # -------- triplanar: world-space --------
        elif mode == "triplanar":
            GL.glUniform1i(self.obj_u_texture_mode, 1)

            # klassisches world-aligned triplanar
            GL.glUniform1f(self.obj_u_triplanar_scale, 0.1)

        # -------- manual triplanar --------
        elif mode == "manual":
            assert (
                mat.texture_scale_value is not None
            ), "texture_scale_value required when texture_scale_mode == 'manual'"

            GL.glUniform1i(self.obj_u_texture_mode, 1)
            GL.glUniform1f(self.obj_u_triplanar_scale, mat.texture_scale_value)

        # -------- safety fallback --------
        else:
            GL.glUniform1i(self.obj_u_texture_mode, 0)
            GL.glUniform1f(self.obj_u_triplanar_scale, 1.0)

        # lighting (dynamic light from world)
        GL.glUniform3f(self.obj_u_light_pos, *self.light_pos)
        GL.glUniform3f(self.obj_u_light_color, *self.light_color)
        GL.glUniform1f(self.obj_u_light_intensity, self.light_intensity)
        GL.glUniform1f(self.obj_u_ambient_strength, 0.25)

        obj.mesh.draw()

    def render_normals(self, objects, camera, aspect):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.gbuffer)
        GL.glUseProgram(self.normal_program)
        GL.glEnable(GL.GL_DEPTH_TEST)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)

        for obj in objects:
            model = obj.transform.matrix()

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.normal_program, "u_view"), 1, GL.GL_TRUE, view
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.normal_program, "u_proj"), 1, GL.GL_TRUE, proj
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.normal_program, "u_model"), 1, GL.GL_TRUE, model
            )

            obj.mesh.draw()

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def render_ssao(self, camera, width, height):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ssao_fbo)
        GL.glUseProgram(self.ssao_program)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_normal)
        GL.glUniform1i(GL.glGetUniformLocation(self.ssao_program, "u_normal"), 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.g_depth)
        GL.glUniform1i(GL.glGetUniformLocation(self.ssao_program, "u_depth"), 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.ssao_noise_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.ssao_program, "u_noise"), 2)

        for i in range(64):
            GL.glUniform3fv(
                GL.glGetUniformLocation(self.ssao_program, f"u_samples[{i}]"),
                1,
                self.ssao_kernel[i],
            )

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.ssao_program, "u_proj"),
            1,
            GL.GL_TRUE,
            camera.get_projection_matrix(width / height),
        )

        GL.glUniform2f(
            GL.glGetUniformLocation(self.ssao_program, "u_noise_scale"),
            width / 4.0,
            height / 4.0,
        )

        GL.glUniform1f(GL.glGetUniformLocation(self.ssao_program, "u_radius"), 0.25)
        GL.glUniform1f(GL.glGetUniformLocation(self.ssao_program, "u_bias"), 0.05)

        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
