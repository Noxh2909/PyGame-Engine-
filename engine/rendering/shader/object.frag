#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;

out vec4 FragColor;

uniform vec3 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform float u_triplanar_scale;

// lighting
uniform vec3 u_light_pos;        // point light (lamp)
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_ambient_strength;

// SSAO
uniform sampler2D u_ssao;
uniform vec2 u_screen_size;
uniform bool u_emissive;

void main()
{
    vec3 baseColor;

    // ---------- texture / color ----------
    if (u_use_texture)
    {
        vec3 n = normalize(vNormal);
        vec3 blend = abs(n);
        blend /= (blend.x + blend.y + blend.z);

        vec3 signN = sign(n);

        vec2 uvX = vec2(vWorldPos.z * signN.x, vWorldPos.y) * u_triplanar_scale;
        vec2 uvY = vec2(vWorldPos.x, vWorldPos.z * signN.y) * u_triplanar_scale;
        vec2 uvZ = vec2(vWorldPos.x * signN.z, vWorldPos.y) * u_triplanar_scale;

        vec4 tx = texture(u_texture, uvX);
        vec4 ty = texture(u_texture, uvY);
        vec4 tz = texture(u_texture, uvZ);

        baseColor = (tx * blend.x + ty * blend.y + tz * blend.z).rgb;
    }
    else
    {
        baseColor = u_color;
    }

    // ---------- emissive (light source itself) ----------
    if (u_emissive)
    {
        FragColor = vec4(baseColor, 1.0);
        return;
    }

    // ---------- lighting ----------
    vec3 n = normalize(vNormal);

    // ambient
    vec3 ambient = u_ambient_strength * baseColor;

    // point light (lamp)
    vec3 lightDir = u_light_pos - vWorldPos;
    float distance = length(lightDir);
    lightDir = normalize(lightDir);

    float diff = max(dot(n, lightDir), 0.0);

    // inverse-square attenuation (stable)
    float attenuation = u_light_intensity / (distance * distance + 1.0);

    vec3 diffuse = diff * baseColor * u_light_color * attenuation;

    // SSAO sampling
    vec2 uv = gl_FragCoord.xy / u_screen_size;
    float ao = clamp(texture(u_ssao, uv).r, 0.25, 1.0);

    vec3 finalColor = (ambient + diffuse) * ao;
    FragColor = vec4(finalColor, 1.0);
}