import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glGenBuffers, glBindVertexArray,
    glBindBuffer, glBufferData, glEnableVertexAttribArray,
    glVertexAttribPointer, glDrawArrays,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES
)
from gameobjects.assets.vertec import cube_vertices

class Mesh:
    def __init__(self, vertices: np.ndarray):
        """
        Docstring für __init__

        :param self: The object itself
        :param vertices: The vertices of the mesh
        :type vertices: np.ndarray
        """
        self.vertex_count = len(vertices) // 3

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.nbytes,
            vertices,
            GL_STATIC_DRAW
        )

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 0, None
        )

        glBindVertexArray(0)

    def draw(self):
        """
        Docstring für draw

        :param self: The object itself
        """
        glBindVertexArray(self.vao)
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
        """
        Docstring für _load_mesh

        :param name: The name of the mesh to load
        :type name: str
        :return: The loaded mesh object
        :rtype: Mesh
        """
        if name == "cube1":
            return Mesh(cube_vertices)
        elif name == "cube2":
            return Mesh(cube_vertices)
        elif name == "cube3":
            return Mesh(cube_vertices)
        elif name == "cube4":
            return Mesh(cube_vertices)
        else:
            raise ValueError(f"Unknown mesh asset: {name}")