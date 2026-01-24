import os
import pygame
import numpy as np
from OpenGL import GL

from rendering.renderer import Renderer, RenderObject
from world import World
from physics.world_physics import PhysicsWorld
from input import InputState

from gameobjects.player.player import Player
from gameobjects.player.camera import Camera
from gameobjects.transform import Transform
from gameobjects.material_lookup import Material
from gameobjects.mesh import Mesh
from gameobjects.glb_loader import GLBLoader
from gameobjects.collider.aabb import AABBCollider
from gameobjects.texture import load_texture
from gameobjects.object import GameObject
from gameobjects.vertec import plane_vertices


# ====================
# Pygame / OpenGL init
# ====================

pygame.init()
pygame.display.set_caption("3D Engine")

pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

WIDTH, HEIGHT = 1400, 800
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
GL.glViewport(0, 0, WIDTH, HEIGHT)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
pygame.mouse.get_rel()

version = GL.glGetString(GL.GL_VERSION)
if version:
    print("OpenGL:", version.decode())


# ====================
# Core engine objects
# ====================

clock = pygame.time.Clock()
input_state = InputState()
physics = PhysicsWorld()
player = Player()
camera = Camera(player, physics)
renderer = Renderer(width=WIDTH, height=HEIGHT)
world = World("engine/world_gen.json")

# ====================
# Static Plane
# ====================

# Create Physics Plane
plane_game_object = GameObject(
    mesh=None,
    transform=Transform(position=(0.0, 0.0, 0.0)),
    material=None,
    collider=AABBCollider(size=(1000.0, 0.0, 1000.0))
)

physics.add_static(plane_game_object)

# Create Render Plane
plane_mesh = Mesh(plane_vertices)

# ====================
# Register world colliders
# ====================

for obj in world.objects:
    if obj.collider is not None:
        physics.add_static(obj)


# ====================
# Sun / Light
# ====================

sun = world.sun
if sun and sun.light:
    renderer.set_light(
        position=sun.transform.position,            
        direction=sun.light["direction"],           
        color=sun.light["color"],
        intensity=sun.light["intensity"],
        ambient=sun.light.get("ambient_strength")
    )

# ====================
# Load mannequin (glTF)
# ====================

loader = GLBLoader("assets/models/idle.glb")
gltf = loader.load_first_mesh()

mannequin_mesh = Mesh(gltf["vertices"], gltf["indices"])

mannequin_material = Material(color=(1.0, 1.0, 1.0))

if gltf["albedo"] is not None:
    temp_dir = "engine/gameobjects/assets/skins"
    os.makedirs(temp_dir, exist_ok=True)
    albedo_path = os.path.join(temp_dir, "mannequin_albedo.png")
    gltf["albedo"].save(albedo_path)

    from gameobjects.texture import load_texture
    mannequin_material.texture = load_texture(albedo_path)


mannequin_render_obj = RenderObject(
    mesh=mannequin_mesh,
    transform=Transform(),
    material=mannequin_material,
)


# ====================
# Scene object list
# ====================

scene_objects: list[RenderObject] = []

for obj in world.objects:
    if obj.mesh is not None:
        scene_objects.append(
            RenderObject(
                mesh=obj.mesh,
                transform=obj.transform,
                material=obj.material,
            )
        )

# Player mannequin is part of the render scene
# scene_objects.append(mannequin_render_obj)


# ====================
# Main Loop
# ====================

running = True
first_person = True
camera.third_person = False

while running:
    dt = clock.tick(240) / 1000.0

    # -------------
    # Events
    # -------------
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_rel()
    player.process_mouse(mx, my)

    actions = input_state.update()

    if actions["toggle_third_person"]:
        first_person = not first_person
        camera.third_person = not first_person

    # -------------
    # Player + Physics
    # -------------
    player.prev_position = player.position.copy()
    player.process_keyboard(actions, dt)
    physics.step(dt, player)

    # -------------
    # Sync mannequin to player
    # -------------
    # mannequin_render_obj.transform.position = player.position.copy()
    # mannequin_render_obj.transform.rotation_y = player.yaw

    # Hide mannequin in first-person
    # mannequin_render_obj.material.visible = camera.third_person

    # -------------
    # Render passes
    # -------------
    light_space = renderer.light_space_matrix(player)

    # Shadow pass
    renderer.render_shadow_pass(
        light_space,
        scene_objects,
    )

    # SSAO pass
    renderer.render_ssao_pass(
        camera,
        scene_objects,
    )

    # Final lighting pass
    renderer.render_final_pass(
        player,
        camera,
        light_space,
        scene_objects,
    )

    # Debug grid (NICHT Teil der Scene)
    GL.glDisable(GL.GL_CULL_FACE)
    renderer.draw_debug_grid(camera, WIDTH / HEIGHT, size=50.0)
    GL.glEnable(GL.GL_CULL_FACE)

    # -------------
    # Debug HUD
    # -------------
    renderer.render_debug_hud(clock, player)
    
    keys = pygame.key.get_pressed()
    
    sphere_speed = 0.1

    if sun is not None:
        if keys[pygame.K_UP]:
            sun.transform.position[2] -= sphere_speed
        if keys[pygame.K_DOWN]:
            sun.transform.position[2] += sphere_speed
        if keys[pygame.K_LEFT]:
            sun.transform.position[0] -= sphere_speed
        if keys[pygame.K_RIGHT]:
            sun.transform.position[0] += sphere_speed
        if keys[pygame.K_PAGEUP]:
            sun.transform.position[1] += sphere_speed
        if keys[pygame.K_PAGEDOWN]:
            sun.transform.position[1] -= sphere_speed
    
    pygame.display.flip()

pygame.quit()