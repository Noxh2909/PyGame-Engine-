import pygame
import numpy as np
import ctypes
from OpenGL.GL import (
    glUseProgram, glGetUniformLocation, glUniform2f,
    glGenVertexArrays, glGenBuffers, glBindVertexArray,
    glBindBuffer, glBufferData, glEnableVertexAttribArray,
    glVertexAttribPointer, glCreateShader, glShaderSource,
    glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glBindTexture,
    glGenTextures, glTexParameteri, glTexImage2D, glActiveTexture,
    glDrawArrays,
    glEnable, glDisable, glBlendFunc,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_TRIANGLES, GL_TEXTURE0,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
)

VERT_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;
uniform vec2 u_offset;   // pixel offset
uniform vec2 u_scale;    // pixel scale
uniform vec2 u_view;     // viewport size
void main() {
    // convert from pixel space to NDC
    vec2 pos = aPos * u_scale + u_offset;
    vec2 ndc = (pos / u_view) * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
    vUV = aUV;
}
"""

FRAG_SRC = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D u_tex;
void main() {
    FragColor = texture(u_tex, vUV);
}
"""

class DebugHUD:
    def __init__(self, viewport_size):
        """
        Docstring für __init__

        :param self: The object itself
        :param viewport_size: The size of the viewport
        """
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 16)
        self.enabled = True
        self.vw, self.vh = viewport_size

        # shader
        self.prog = self._make_program(VERT_SRC, FRAG_SRC)
        self.u_offset = glGetUniformLocation(self.prog, "u_offset")
        self.u_scale  = glGetUniformLocation(self.prog, "u_scale")
        self.u_view   = glGetUniformLocation(self.prog, "u_view")

        # quad (two triangles)
        verts = np.array([
            0, 0, 0, 1,
            1, 0, 1, 1,
            1, 1, 1, 0,
            0, 0, 0, 1,
            1, 1, 1, 0,
            0, 1, 0, 0,
        ], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        # texture
        self.tex = glGenTextures(1)

    def _make_program(self, vs, fs):
        """
        Docstring für _make_program

        :param self: The object itself
        :param vs: The vertex shader source
        :param fs: The fragment shader source
        """
        def sh(src, t):
            s = glCreateShader(t)
            glShaderSource(s, src)
            glCompileShader(s)
            if not glGetShaderiv(s, GL_COMPILE_STATUS):
                raise RuntimeError(glGetShaderInfoLog(s))
            return s
        p = glCreateProgram()
        glAttachShader(p, sh(vs, GL_VERTEX_SHADER))
        glAttachShader(p, sh(fs, GL_FRAGMENT_SHADER))
        glLinkProgram(p)
        if not glGetProgramiv(p, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(p))
        return p

    def _upload_surface(self, surf):
        """
        Docstring für _upload_surface

        :param self: The object itself
        :param surf: The surface to upload
        """
        data = pygame.image.tostring(surf, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            surf.get_width(), surf.get_height(),
            0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )
        return surf.get_width(), surf.get_height()

    def draw(self, clock, camera):
        if not self.enabled:
            return

        pos = camera.transform.position
        lines = [
            f"FPS: {clock.get_fps():.1f}",
            f"Pos: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}",
        ]

        glUseProgram(self.prog)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUniform2f(self.u_view, float(self.vw), float(self.vh))
        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        x, y = 10, 10
        for line in lines:
            surf = self.font.render(line, True, (255, 255, 255)).convert_alpha()
            w, h = self._upload_surface(surf)
            glUniform2f(self.u_offset, float(x), float(y))
            glUniform2f(self.u_scale, float(w), float(h))
            glDrawArrays(GL_TRIANGLES, 0, 6)
            y += h + 4

        glBindVertexArray(0)
        glDisable(GL_BLEND)
        glUseProgram(0)