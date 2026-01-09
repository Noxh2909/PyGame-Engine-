#version 330 core

out float FragColor;
in vec2 vUV;
uniform sampler2D u_ssao;

void main()
{
    float result = 0.0;
    vec2 texel = 1.0 / textureSize(u_ssao, 0);

    for (int x = -2; x <= 2; x++)
    for (int y = -2; y <= 2; y++)
        result += texture(u_ssao, vUV + vec2(x, y) * texel).r;

    FragColor = result / 25.0;
}