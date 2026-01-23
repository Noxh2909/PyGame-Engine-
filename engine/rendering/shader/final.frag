#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace;
in vec4 ReflClipPos;
uniform sampler2D shadowMap;
uniform sampler2D ssaoTexture;
// Planar reflection texture (rendered from mirrored camera)
uniform sampler2D reflectionTex;
// 0.0 = no reflection, 1.0 = full reflection
uniform float reflectivity;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform sampler2D u_texture;
uniform bool u_use_texture;
uniform bool u_is_emissive;
uniform int u_texture_mode; // 0 = UV, 1 = triplanar
uniform float u_lightIntensity;
uniform float u_ambientStrength;
uniform float u_triplanar_scale;

float ShadowCalculation(vec4 fragPosLightSpace) {
  vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
  projCoords = projCoords * 0.5 + 0.5;

  if (projCoords.z > 1.0)
    return 0.0;

  float shadow = 0.0;
  vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
  float bias =
      max(0.005 * (1.0 - dot(normalize(Normal), normalize(lightPos - FragPos))),
          0.0005);

  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y) {
      float pcfDepth =
          texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;

      shadow += (projCoords.z - bias > pcfDepth) ? 1.0 : 0.0;
    }

  return shadow / 9.0;
}

void main() {
  vec3 baseColor;

  if (u_is_emissive) {
    FragColor = vec4(objectColor, 1.0);
    return;
  }

  if (u_use_texture) {
    if (u_texture_mode == 0) {
      // UV mapping
      baseColor = texture(u_texture, TexCoord).rgb;
    } else {
      // Triplanar mapping (world-space)
      vec3 n = normalize(Normal);
      vec3 blend = abs(n);
      blend = max(blend, vec3(0.0001));
      blend /= (blend.x + blend.y + blend.z);

      vec2 uvX = FragPos.zy * u_triplanar_scale;
      vec2 uvY = FragPos.xz * u_triplanar_scale;
      vec2 uvZ = FragPos.xy * u_triplanar_scale;

      vec3 tx = texture(u_texture, uvX).rgb;
      vec3 ty = texture(u_texture, uvY).rgb;
      vec3 tz = texture(u_texture, uvZ).rgb;

      baseColor = tx * blend.x + ty * blend.y + tz * blend.z;
    }
  } else {
    baseColor = objectColor;
  }

  vec2 screenUV = gl_FragCoord.xy / vec2(textureSize(ssaoTexture, 0));
  float ao = texture(ssaoTexture, screenUV).r;
  vec3 ambient = u_ambientStrength * ao * lightColor * u_lightIntensity;
  vec3 norm = normalize(Normal);
  vec3 lightDir = normalize(lightPos - FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor * baseColor * u_lightIntensity;
  vec3 viewDir = normalize(viewPos - FragPos);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  vec3 specular = spec * lightColor * u_lightIntensity;
  float shadow = ShadowCalculation(FragPosLightSpace);

  // Add attenuation based on distance
  float distance = length(lightPos - FragPos);
  float attenuation =
      1.0 / (1.0 + 0.09 * distance +
             0.032 * distance * distance); // Quadratic attenuation
  vec3 lighting = ambient + attenuation * (1.0 - shadow) *
                                (diffuse + specular); // Apply shadow
  FragColor = vec4(lighting * baseColor, 1.0);
}