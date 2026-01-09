import numpy as np

cube_vertices = np.array([
    # Front (+Z)
    -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,

    # Back (-Z)
     0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
     0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, -1.0,

    # Left (-X)
    -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,
    -0.5, -0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,  -1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,

    # Right (+X)
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,

    # Top (+Y)
    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,

    # Bottom (-Y)
    -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
], dtype=np.float32)


# Sphere generated with latitude/longitude tessellation
# Vertex format: x, y, z, nx, ny, nz
def generate_sphere(radius=0.5, stacks=16, slices=32):
    verts = []
    for i in range(stacks):
        phi1 = np.pi * i / stacks
        phi2 = np.pi * (i + 1) / stacks

        for j in range(slices):
            theta1 = 2 * np.pi * j / slices
            theta2 = 2 * np.pi * (j + 1) / slices

            # four points on the sphere
            p1 = np.array([
                np.sin(phi1) * np.cos(theta1),
                np.cos(phi1),
                np.sin(phi1) * np.sin(theta1)
            ])
            p2 = np.array([
                np.sin(phi2) * np.cos(theta1),
                np.cos(phi2),
                np.sin(phi2) * np.sin(theta1)
            ])
            p3 = np.array([
                np.sin(phi2) * np.cos(theta2),
                np.cos(phi2),
                np.sin(phi2) * np.sin(theta2)
            ])
            p4 = np.array([
                np.sin(phi1) * np.cos(theta2),
                np.cos(phi1),
                np.sin(phi1) * np.sin(theta2)
            ])

            # two triangles per quad
            for a, b, c in [(p1, p2, p3), (p1, p3, p4)]:
                for v in (a, b, c):
                    pos = v * radius
                    nrm = v / np.linalg.norm(v)
                    verts.extend([pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2]])

    return np.array(verts, dtype=np.float32)


sphere_vertices = generate_sphere(radius=0.5, stacks=16, slices=32)

cylinder_vertices = np.array([
    # Placeholder for cylinder vertices
], dtype=np.float32)