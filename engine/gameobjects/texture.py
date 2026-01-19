import pygame
from OpenGL.GL import (
    glGenTextures,
    glBindTexture,
    glTexImage2D,
    glTexParameteri,
    glGenerateMipmap,
    GL_TEXTURE_2D,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
)


def load_texture(path: str) -> int:
    """
    Load a PNG/JPG texture from disk and upload it to OpenGL.
    Returns the OpenGL texture ID.
    """

    # Load image via pygame
    surface = pygame.image.load(path).convert_alpha()
    width, height = surface.get_size()

    # Convert to raw bytes (RGBA)
    image_data = pygame.image.tostring(surface, "RGBA", True)

    # Create OpenGL texture
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image_data,
    )

    # Texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id
