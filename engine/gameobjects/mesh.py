import numpy as np
import ctypes
from OpenGL.GL import (
    glGenVertexArrays, glGenBuffers, glBindVertexArray,
    glBindBuffer, glBufferData, glEnableVertexAttribArray,
    glVertexAttribPointer, glDrawArrays, glDrawElements,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES, GL_UNSIGNED_INT
)
from gameobjects.assets.vertec import cube_vertices, sphere_vertices

class Mesh:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray | None = None):
        """
        vertices: interleaved array [pos(3), normal(3), uv(2)]
        indices: optional index buffer (uint32)
        """
        self.vertex_count = len(vertices) // 8
        self.index_count = len(indices) if indices is not None else 0
        self.has_indices = indices is not None

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1) if self.has_indices else None

        glBindVertexArray(self.vao)

        # VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # EBO (optional)
        if self.has_indices and indices is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER,
                indices.nbytes,
                indices.astype(np.uint32),
                GL_STATIC_DRAW,
            )

        stride = 8 * 4  # 8 floats * 4 bytes

        # position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # normal (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))

        # uv (location = 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        if self.has_indices:
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)
        
class MeshRegistry:
    """
    Docstring für MeshRegistry
    """
    _meshes: dict[str, Mesh] = {}

    @classmethod
    def get(cls, name: str) -> Mesh:
        """
        Docstring für get

        :param cls: The class itself
        :param name: The name of the mesh to retrieve
        :type name: str
        :return: The mesh object
        :rtype: Mesh
        """
        if name not in cls._meshes:
            cls._meshes[name] = cls._load_mesh(name)
        return cls._meshes[name]

    @staticmethod
    def _load_mesh(name: str) -> Mesh:
        if name == "cube":
            return Mesh(cube_vertices)
        elif name == "sphere":
            return Mesh(sphere_vertices)
        else:
            raise ValueError(f"Unknown mesh asset: {name}")