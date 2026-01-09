#version 330 core

out float FragColor;
in vec2 vUV;

uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform sampler2D u_noise;

uniform vec3 u_samples[64];
uniform mat4 u_proj;

uniform vec2 u_noise_scale;
uniform float u_radius;
uniform float u_bias;

void main()
{
    vec3 normal = normalize(texture(u_normal, vUV).xyz * 2.0 - 1.0);
    float depth = texture(u_depth, vUV).r;

    vec3 randomVec = texture(u_noise, vUV * u_noise_scale).xyz;
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;

    for (int i = 0; i < 64; i++)
    {
        vec3 samplePos = TBN * u_samples[i];
        samplePos = samplePos * u_radius + vec3(0.0, 0.0, depth);

        vec4 offset = vec4(samplePos, 1.0);
        offset = u_proj * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(u_depth, offset.xy).r;
        float rangeCheck = smoothstep(0.0, 1.0, u_radius / abs(depth - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + u_bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / 64.0);
    FragColor = occlusion;
}
