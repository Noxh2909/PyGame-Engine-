import pygame
from OpenGL.GL import (
    glClear, glClearColor,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glViewport, glGetString, GL_VERSION,
    glEnable, GL_DEPTH_TEST
)

from camera import FPSCamera
from rendering.renderer import Renderer
from input import InputState
from debug import DebugHUD
from physics import PhysicsWorld
from world import World
from gameobjects.object import GameObject
from gameobjects.transform import Transform
from gameobjects.collider.aabb import AABBCollider

# --------------------
# Pygame / OpenGL setup
# --------------------

pygame.init()
pygame.display.set_caption("3D Engine")

# OpenGL 3.3 Core
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK,
    pygame.GL_CONTEXT_PROFILE_CORE
)

width, height = 1400, 800
pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
glViewport(0, 0, width, height)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

version = glGetString(GL_VERSION)
if version:
    print("OpenGL:", version.decode())

glEnable(GL_DEPTH_TEST)

# --------------------
# Engine objects
# --------------------

camera = FPSCamera()
input_state = InputState()
renderer = Renderer()
debug = DebugHUD((width, height))
physics = PhysicsWorld()
world = World("engine/world_gen.json")

# --------------------
# Hardcoded world objects
# --------------------

ground_plane = GameObject(
    mesh=None,              # no rendering mesh
    material=None,          # no material
    transform=Transform(position=(0, 0, 0), scale=(1000, 1.0, 1000)),
    collider=AABBCollider(size=(1000, 1.0, 1000))
)

physics.add_static(ground_plane)

sun = world.sun
if sun is not None and sun.light is not None:
    renderer.set_light(
        position=sun.transform.position,
        color=sun.light.get("color"),
        intensity=sun.light.get("intensity")
    )

# --------------------  
# Visual cube ONLY
# --------------------

for obj in world.objects:
    if obj.collider is not None:
        physics.add_static(obj)

clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(240) / 1000.0

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_rel()
    camera.process_mouse(mx, my)

    actions = input_state.update()

    # store previous position for physics (ground / wall detection)
    camera.prev_position = camera.position.copy()

    camera.process_keyboard(actions, dt)
    physics.step(dt, camera)

    # ---------- SSAO Phase A: Normal + Depth pass ----------
    renderer.render_normals(world.objects, camera, width / height)
    renderer.render_ssao(camera, width, height)

    # ---------- Normal render pass ----------
    glClearColor(0.05, 0.05, 0.08, 1.0)
    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))

    renderer.draw_plane(camera, width / height)
    for obj in world.objects:
        if obj.mesh is not None:
            renderer.draw_object(obj, camera, width / height)

    debug.draw(clock, camera)
    pygame.display.flip()

pygame.quit()