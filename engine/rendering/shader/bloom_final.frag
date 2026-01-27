#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D hdrScene;
uniform sampler2D bloomBlur;
uniform float bloomIntensity;

void main() {
  vec3 hdr = texture(hdrScene, TexCoord).rgb;
  vec3 bloom = texture(bloomBlur, TexCoord).rgb;

  // Bloom nur subtil addieren (kein Nebel!)
  vec3 color = hdr + bloom * bloomIntensity;

  FragColor = vec4(color, 1.0);
}