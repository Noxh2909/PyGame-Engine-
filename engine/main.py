import os
import pygame
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
from gameobjects.texture import Texture
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
    transform=Transform(position=(0.0, 0.05, 0.0)),
    material=None,
    collider=AABBCollider(size=(1000.0, 0.1, 1000.0))
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
    temp_dir = "assets/skins"
    os.makedirs(temp_dir, exist_ok=True)
    albedo_path = os.path.join(temp_dir, "mannequin_albedo.png")
    gltf["albedo"].save(albedo_path)

    from gameobjects.texture import Texture
    mannequin_material.texture = Texture.load_texture(albedo_path)


mannequin_render_obj = RenderObject(
    mesh=mannequin_mesh,
    transform=Transform(scale=(2.8, 2.8, 2.8)),
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
scene_objects.append(mannequin_render_obj)


# ====================
# Main Loop
# ====================

running = True
first_person = True
camera.third_person = False

# State for object control
control_state = {
    'target': 'sun',
    'm_was_pressed': False
}

while running:
    dt = clock.tick(120) / 1000.0

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

    # -------------
    # Render passes
    # -------------
    light_space_matrix = renderer.point_light_matrices()

    # Shadow pass
    renderer.render_shadow_pass(scene_objects)

    # SSAO pass
    renderer.render_ssao_pass(camera, scene_objects)

    # Final lighting pass
    renderer.render_final_pass(player, camera, scene_objects)
    
    # Debug grid
    renderer.draw_debug_grid(camera, WIDTH / HEIGHT, size=50.0)
    
    # Bloom pass
    renderer.render_bloom_pass()
    
    # -------------
    # DEBUG OBJECT CONTROL
    # -------------
    
    keys = pygame.key.get_pressed()
    
    obj_movement_speed = 0.1

    # Toggle control target with 'M' key (single press)
    if keys[pygame.K_m] and not control_state['m_was_pressed']:
        # Get list of controllable objects
        controllable_objects = ['sun', 'mannequin'] + [f'scene_{i}' for i in range(len(scene_objects) - 1)]
        current_index = controllable_objects.index(control_state['target'])
        next_index = (current_index + 1) % len(controllable_objects)
        control_state['target'] = controllable_objects[next_index]
        control_state['m_was_pressed'] = True
    elif not keys[pygame.K_m]:
        control_state['m_was_pressed'] = False
    
    control_target = control_state['target']
    
    # Determine which object to control
    target_transform = None
    if control_target == "mannequin":
        target_transform = mannequin_render_obj.transform
    elif control_target == "sun" and sun is not None:
        target_transform = sun.transform
    elif control_target.startswith("scene_"):
        scene_index = int(control_target.split('_')[1])
        if scene_index < len(scene_objects) - 1:  # Exclude mannequin
            target_transform = scene_objects[scene_index].transform
    
    # Get object position
    object_position = None
    if target_transform is not None:
        object_position = target_transform.position
    
    object_scale = None 
    if target_transform is not None:
        object_scale = target_transform.scale
    
    # Apply movement to target object
    if target_transform is not None:
        if keys[pygame.K_UP]:
            target_transform.position[2] -= obj_movement_speed
        if keys[pygame.K_DOWN]:
            target_transform.position[2] += obj_movement_speed
        if keys[pygame.K_LEFT]:
            target_transform.position[0] -= obj_movement_speed
        if keys[pygame.K_RIGHT]:
            target_transform.position[0] += obj_movement_speed
        if keys[pygame.K_PAGEUP]:
            target_transform.position[1] += obj_movement_speed
        if keys[pygame.K_PAGEDOWN]:
            target_transform.position[1] -= obj_movement_speed
            
    # Debug HUD
    renderer.render_debug_hud(clock, player, obj=control_state, obj_pos=object_position, obj_scale=object_scale)
    
    
    pygame.display.flip()

pygame.quit()