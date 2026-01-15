#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;
layout (location = 3) in ivec4 aJoints;
layout (location = 4) in vec4 aWeights;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_bones[64];   // MAX_BONES

void main()
{
    mat4 skin =
        aWeights.x * u_bones[aJoints.x] +
        aWeights.y * u_bones[aJoints.y] +
        aWeights.z * u_bones[aJoints.z] +
        aWeights.w * u_bones[aJoints.w];

    vec4 skinnedPos = skin * vec4(aPos, 1.0);
    vec4 world = u_model * skinnedPos;

    vWorldPos = world.xyz;

    mat3 normalMatrix = mat3(transpose(inverse(u_model * skin)));
    vNormal = normalize(normalMatrix * aNormal);

    vUV = aUV;

    gl_Position = u_proj * u_view * world;
}