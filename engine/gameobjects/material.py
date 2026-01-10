import os

class Material:
    def __init__(self, color=(1.0, 1.0, 1.0), texture=None, emissive=False, texture_scale_mode=None, texture_scale_value=None):
        """
        color         : fallback color (vec3)
        texture       : OpenGL texture id or None
        emissive      : bool
        texture_scale_mode : optional scale mode for texture coordinates
        texture_scale_value : optional scale value for texture coordinates
        """
        self.color = color
        self.texture = texture
        self.emissive = emissive
        self.texture_scale_mode = texture_scale_mode
        self.texture_scale_value = texture_scale_value