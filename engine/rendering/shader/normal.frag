#version 330 core

in vec3 vNormal;
out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal) * 0.5 + 0.5;
    FragColor = vec4(n, 1.0);
}
