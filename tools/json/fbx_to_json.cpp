#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>

#define UFBX_IMPLEMENTATION
#include "../ufbx/ufbx.h"

// nlohmann/json single-header
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static std::array<float, 16> flatten_mat4(const ufbx_matrix &m)
{
    // ufbx_matrix is 3x4, expand to 4x4 (column-major)
    return {
        (float)m.m00, (float)m.m10, (float)m.m20, 0.0f,
        (float)m.m01, (float)m.m11, (float)m.m21, 0.0f,
        (float)m.m02, (float)m.m12, (float)m.m22, 0.0f,
        (float)m.m03, (float)m.m13, (float)m.m23, 1.0f,
    };
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: fbx_to_json <file.fbx>\n");
        return 1;
    }

    const char *path = argv[1];

    std::string input_path = path;
    std::string base = input_path;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }

    std::string mesh_path      = base + "_mesh.json";
    std::string skin_path      = base + "_skin.json";
    std::string skeleton_path  = base + "_skeleton.json";
    std::string animation_path = base + "_animation.json";

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(path, nullptr, &error);
    if (!scene) {
        fprintf(stderr, "ufbx load failed: %s\n", error.description.data);
        return 1;
    }

    json mesh_root;
    json skin_root;
    json skeleton_root;
    json animation_root;

    json meta = {
        {"generator", "ufbx"},
        {"source", path},
        {"version", 1},
        {"unit_scale", 1.0}
    };

    mesh_root["meta"]      = meta;
    skin_root["meta"]      = meta;
    skeleton_root["meta"]  = meta;
    animation_root["meta"] = meta;

    // ------------------------------------------------------------
    // Skeleton
    // ------------------------------------------------------------
    json bones = json::array();
    std::vector<int> node_to_bone(scene->elements.count, -1);

    int bone_index = 0;
    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *node = scene->nodes.data[i];
        if (!node->bone) continue;

        node_to_bone[node->element_id] = bone_index;

        int parent = -1;
        if (node->parent) {
            int pid = node->parent->element_id;
            if (pid >= 0 && pid < (int)node_to_bone.size()) {
                parent = node_to_bone[pid];
            }
        }

        bones.push_back({
            {"index", bone_index},
            {"name", std::string(node->name.data)},
            {"parent", parent},
            {"inverse_bind", flatten_mat4(ufbx_identity_matrix)}
        });

        bone_index++;
    }

    skeleton_root["bones"] = bones;

    // ------------------------------------------------------------
    // Mesh + Skinning (INDEX-BASED, ufbx-correct)
    // ------------------------------------------------------------
    if (scene->meshes.count > 0) {
        ufbx_mesh *mesh = scene->meshes.data[0];

        json vertices = json::array();
        json indices  = json::array();
        json joints   = json::array();
        json weights  = json::array();

        std::vector<std::array<int,4>> vertex_joints(mesh->num_vertices, {0,0,0,0});
        std::vector<std::array<float,4>> vertex_weights(mesh->num_vertices, {0,0,0,0});

        if (mesh->skin_deformers.count > 0) {
            ufbx_skin_deformer *skin = mesh->skin_deformers.data[0];

            for (size_t c = 0; c < skin->clusters.count; c++) {
                ufbx_skin_cluster *cluster = skin->clusters.data[c];

                int bone = 0;
                int eid = cluster->bone_node->element_id;
                if (eid >= 0 && eid < (int)node_to_bone.size()) {
                    bone = node_to_bone[eid];
                }

                for (size_t wi = 0; wi < cluster->weights.count; wi++) {
                    uint32_t v = cluster->vertices.data[wi];
                    float w = (float)cluster->weights.data[wi];

                    auto &vj = vertex_joints[v];
                    auto &vw = vertex_weights[v];

                    for (int k = 0; k < 4; k++) {
                        if (vw[k] == 0.0f) {
                            vj[k] = bone;
                            vw[k] = w;
                            break;
                        }
                    }
                }
            }

            for (size_t v = 0; v < mesh->num_vertices; v++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += vertex_weights[v][k];
                if (sum > 0.0f) {
                    for (int k = 0; k < 4; k++) vertex_weights[v][k] /= sum;
                }
            }
        }

        for (size_t i = 0; i < mesh->num_indices; i++) {

            // index -> logical vertex
            uint32_t vi = mesh->vertex_indices.data[i];

            // vertex attributes are INDEXED, not vertex-based
            ufbx_vec3 pos = mesh->vertex_position.values.data[
                mesh->vertex_position.indices.data[i]
            ];

            ufbx_vec3 nor = mesh->vertex_normal.exists
                ? mesh->vertex_normal.values.data[
                    mesh->vertex_normal.indices.data[i]
                  ]
                : ufbx_vec3{0,0,1};

            ufbx_vec2 uv = mesh->vertex_uv.exists
                ? mesh->vertex_uv.values.data[
                    mesh->vertex_uv.indices.data[i]
                  ]
                : ufbx_vec2{0,0};

            vertices.push_back(pos.x);
            vertices.push_back(pos.y);
            vertices.push_back(pos.z);

            vertices.push_back(nor.x);
            vertices.push_back(nor.y);
            vertices.push_back(nor.z);

            vertices.push_back(uv.x);
            vertices.push_back(uv.y);

            auto &j = vertex_joints[vi];
            auto &w = vertex_weights[vi];

            joints.push_back({j[0], j[1], j[2], j[3]});
            weights.push_back({w[0], w[1], w[2], w[3]});

            indices.push_back((int)i);
        }

        mesh_root["mesh"] = {
            {"vertex_count", (int)mesh->num_indices},
            {"vertices", vertices},
            {"indices", indices}
        };

        skin_root["skin"] = {
            {"vertex_count", (int)mesh->num_indices},
            {"joints", joints},
            {"weights", weights}
        };
    }

    // ------------------------------------------------------------
    // Animations (baked)
    // ------------------------------------------------------------
    if (scene->anim_stacks.count > 0) {
        json animations = json::object();

        ufbx_error err;
        ufbx_bake_opts opts = {};
        opts.trim_start_time = true;
        opts.resample_rate = 30.0;

        for (size_t a = 0; a < scene->anim_stacks.count; a++) {
            ufbx_anim_stack *stack = scene->anim_stacks.data[a];

            ufbx_baked_anim *baked = ufbx_bake_anim(scene, stack->anim, &opts, &err);
            if (!baked) continue;

            json clip;
            clip["duration"] = baked->playback_duration;

            json channels = json::array();

            for (size_t i = 0; i < baked->nodes.count; i++) {
                ufbx_baked_node &bn = baked->nodes.data[i];

                if (bn.element_id >= scene->elements.count) continue;
                ufbx_element *elem = scene->elements.data[bn.element_id];
                if (!elem || elem->type != UFBX_ELEMENT_NODE) continue;

                ufbx_node *node = (ufbx_node*)elem;

                int bone = -1;
                int eid = node->element_id;
                if (eid >= 0 && eid < (int)node_to_bone.size()) {
                    bone = node_to_bone[eid];
                }
                if (bone < 0) continue;

                json channel;
                channel["bone"] = bone;

                if (bn.translation_keys.count > 0) {
                    json t;
                    json times = json::array();
                    json values = json::array();
                    for (size_t k = 0; k < bn.translation_keys.count; k++) {
                        times.push_back(bn.translation_keys.data[k].time);
                        auto &v = bn.translation_keys.data[k].value;
                        values.push_back({v.x, v.y, v.z});
                    }
                    t["times"] = times;
                    t["values"] = values;
                    channel["translation"] = t;
                }

                if (bn.rotation_keys.count > 0) {
                    json r;
                    json times = json::array();
                    json values = json::array();
                    for (size_t k = 0; k < bn.rotation_keys.count; k++) {
                        times.push_back(bn.rotation_keys.data[k].time);
                        auto &q = bn.rotation_keys.data[k].value;
                        values.push_back({q.x, q.y, q.z, q.w});
                    }
                    r["times"] = times;
                    r["values"] = values;
                    channel["rotation"] = r;
                }

                if (bn.scale_keys.count > 0) {
                    json s;
                    json times = json::array();
                    json values = json::array();
                    for (size_t k = 0; k < bn.scale_keys.count; k++) {
                        times.push_back(bn.scale_keys.data[k].time);
                        auto &v = bn.scale_keys.data[k].value;
                        values.push_back({v.x, v.y, v.z});
                    }
                    s["times"] = times;
                    s["values"] = values;
                    channel["scale"] = s;
                }

                channels.push_back(channel);
            }

            clip["channels"] = channels;

            std::string name = stack->name.length > 0
                ? std::string(stack->name.data)
                : ("anim_" + std::to_string(a));

            animations[name] = clip;

            ufbx_free_anim((ufbx_anim*)baked);
        }

        animation_root["animations"] = animations;
    }

    auto write_json = [](const std::string &p, const json &j) {
        FILE *f = fopen(p.c_str(), "w");
        if (f) {
            fprintf(f, "%s\n", j.dump(2).c_str());
            fclose(f);
        }
    };

    write_json(mesh_path, mesh_root);
    write_json(skin_path, skin_root);
    write_json(skeleton_path, skeleton_root);
    write_json(animation_path, animation_root);

    printf("Wrote:\n  %s\n  %s\n  %s\n  %s\n",
        mesh_path.c_str(),
        skin_path.c_str(),
        skeleton_path.c_str(),
        animation_path.c_str()
    );

    ufbx_free_scene(scene);
    return 0;
}