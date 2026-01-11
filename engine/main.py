import pygame
from OpenGL.GL import (
    glClear, glClearColor,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glViewport, glGetString, GL_VERSION,
    glEnable, GL_DEPTH_TEST
)

from gameobjects.material import Material
from gameobjects.texture import load_texture
import os

# from gameobjects.player.mannequin.capsule_mannequin import CapsuleMannequin, CapsuleBodyMesh, CapsuleHeadMesh
from gameobjects.assets.vertec import cylinder_vertices, sphere_vertices
from gameobjects.mesh import Mesh  # adjust if your Mesh class lives elsewhere

from gameobjects.player.mannequin.static_mannequin import StaticMannequin
from gameobjects.player.mannequin.capsule_mannequin import CapsuleMannequin, CapsuleHeadMesh, CapsuleBodyMesh

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
from gameobjects.assets.glb_loader import load_gltf_mesh

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
# Static mannequin (glTF / .glb)
# --------------------

vertices, indices, albedo_image = load_gltf_mesh(
    "engine/gameobjects/assets/models/basic_player.glb"
)

mannequin_mesh = Mesh(vertices, indices)
mannequin_height = 1.8  # reale Körperhöhe

# --- create material (upload albedo texture) ---
mannequin_material = Material(color=(1.0, 1.0, 1.0))

if albedo_image is not None:
    temp_dir = "engine/gameobjects/assets/_tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "mannequin_albedo.png")

    albedo_image.save(temp_path)
    mannequin_material.texture = load_texture(temp_path)

# --- create mannequin ---
static_mannequin = StaticMannequin(
    player=player,
    body_mesh=mannequin_mesh,
    height=mannequin_height,
)

# attach material
static_mannequin.material = mannequin_material

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
first_person = True 
camera.third_person = False

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
    player.prev_position = player.position.copy()

    player.process_keyboard(actions, dt)
    physics.step(dt, player)

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
    # Draw player mannequin (third-person / debug)
    # --------------------

    # if not first_person and capsule_mesh is not None:
    #     mannequin.draw(renderer.object_program)
    
    if not first_person:
        renderer.draw_object(static_mannequin, camera, width / height)

    debug.draw(clock, player)
    pygame.display.flip()

pygame.quit()