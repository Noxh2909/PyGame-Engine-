#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    vec4 world = u_model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;

    // transform normal to world space
    mat3 normalMatrix = mat3(transpose(inverse(u_model)));
    vNormal = normalize(normalMatrix * aNormal);

    vUV = aUV;

    gl_Position = u_proj * u_view * world;
}
