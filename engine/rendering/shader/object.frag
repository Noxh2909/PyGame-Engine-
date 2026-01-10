#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

out vec4 FragColor;

uniform vec3 u_color;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform int u_texture_mode;
// 0 = default (UV), 1 = triplanar
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
        // -------- default: UV mapping (object-local) --------
        if (u_texture_mode == 0)
        {
            baseColor = texture(u_texture, vUV).rgb;
        }
        // -------- triplanar: world-space --------
        else
        {
            vec3 n = normalize(vNormal);
            vec3 blend = abs(n);
            blend = max(blend, vec3(0.0001));
            blend /= (blend.x + blend.y + blend.z);

            vec2 uvX = vWorldPos.zy * u_triplanar_scale;
            vec2 uvY = vWorldPos.xz * u_triplanar_scale;
            vec2 uvZ = vWorldPos.xy * u_triplanar_scale;

            vec3 tx = texture(u_texture, uvX).rgb;
            vec3 ty = texture(u_texture, uvY).rgb;
            vec3 tz = texture(u_texture, uvZ).rgb;

            baseColor = tx * blend.x + ty * blend.y + tz * blend.z;
        }
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