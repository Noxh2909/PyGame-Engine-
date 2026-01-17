import pygame
import math
import numpy as np
from OpenGL.GL import (
    glClear, glClearColor,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glViewport, glGetString, GL_VERSION,
    glEnable, GL_DEPTH_TEST
)

from gameobjects.material_lookup import Material
from gameobjects.texture import load_texture
import os

from gameobjects.mesh import Mesh  # adjust if your Mesh class lives elsewhere

from gameobjects.player.mannequin.mannequin import Mannequin

from rendering.renderer import Renderer
from input import InputState
from debug import DebugHUD
from physics.world_physics import PhysicsWorld
from world import World
from gameobjects.object import GameObject
from gameobjects.transform import Transform
from gameobjects.collider.aabb import AABBCollider
from gameobjects.player.player import Player
from gameobjects.player.camera import Camera

from assets.animations.json_loader import FBXJsonLoader
from assets.animations.skeleton import Skeleton

# NOTE:
# *_mannequin.json contains ONLY skeleton + skinning data (no geometry).
# Geometry is loaded separately and combined here.

# --------------------
# Pygame / OpenGL setup
# --------------------

pygame.init()
pygame.display.set_caption("3D Engine")

# OpenGL 3.3 Corej
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
pygame.mouse.get_rel()  

version = glGetString(GL_VERSION)
if version:
    print("OpenGL:", version.decode())

glEnable(GL_DEPTH_TEST)

# --------------------
# Engine objects
# --------------------

player = Player()
input_state = InputState()
renderer = Renderer()
debug = DebugHUD((width, height))
physics = PhysicsWorld()
camera = Camera(player, physics)
world = World("engine/world_gen.json")

# --------------------
# Character from FBX → JSON
# --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "models")

loader = FBXJsonLoader()

mesh_data = loader.load_mesh(
    os.path.join(ASSETS_DIR, "Punching_mesh.json")
)

skin_data = loader.load_skin(
    os.path.join(ASSETS_DIR, "Punching_skin.json")
)

skeleton_data = loader.load_skeleton(
    os.path.join(ASSETS_DIR, "Punching_skeleton.json")
)

animations = loader.load_animations(
    os.path.join(ASSETS_DIR, "Punching_animation.json")
)

# --------------------
# Apply skinning data from mannequin JSON
# --------------------
vertices = np.array(mesh_data["vertices"], dtype=np.float32)
# Model units already converted during FBX export
# vertices *= 0.01
indices  = np.array(mesh_data["indices"], dtype=np.uint32)
joints   = np.array(skin_data["joints"], dtype=np.int32)
weights  = np.array(skin_data["weights"], dtype=np.float32)

mannequin_mesh = Mesh(
    vertices=vertices,
    indices=indices,
    joints=joints,
    weights=weights,
)

mannequin_height = 1.8

# Skeleton
parents = np.array(skeleton_data["parents"], dtype=np.int32)
inverse_bind = np.array(skeleton_data["inverse_bind"], dtype=np.float32)

skeleton = Skeleton(parents, inverse_bind)
if not hasattr(skeleton, "animations"):
    skeleton.animations = {}

# register animations (no playback yet)
for name, anim in animations.items():
    skeleton.animations[name] = anim

# --- create material (upload albedo texture) ---
mannequin_material = Material(color=(1.0, 1.0, 1.0))

# --- create mannequin ---
mannequin = Mannequin(
    player=player,
    body_mesh=mannequin_mesh,
    height=mannequin_height,
)

mannequin.is_skinned = True
mannequin.skeleton = skeleton

# attach material
mannequin.material = mannequin_material

# --------------------
# Hardcoded world objects
# --------------------

ground_plane = GameObject(
    mesh=None,              # no rendering mesh
    material=None,          # no material
    transform=Transform(position=(0, 0, 0), scale=(100000, 0.0, 100000)),
    collider=AABBCollider(size=(1000, 0.0, 1000))
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

# view mode
first_person = True
camera.third_person = False  # camera follows player in TPS

# FPS/TPS visibility control
# FPS  -> body hidden
# TPS  -> full body visible

while running:
    dt = clock.tick(240) / 1000.0

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_rel()
    player.process_mouse(mx, my)

    actions = input_state.update()
    
    if actions["toggle_third_person"]:
        first_person = not first_person
        camera.third_person = not first_person

    # store previous position for physics (ground / wall detection)
    player.prev_position = player.transform.position.copy()

    player.process_keyboard(actions, dt)
    physics.step(dt, player)

    # --- Skeleton update ---
    skeleton.reset_pose()

    # sync mannequin root with player (BODY follows yaw only)
    # sync mannequin with player (position + yaw)
    mannequin.update_from_player(player)

    skeleton.update()

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

    # --------------------
    # Draw player mannequin
    # --------------------
    renderer.draw_object(mannequin, camera, width / height)

    debug.draw(clock, player)
    pygame.display.flip()

pygame.quit()