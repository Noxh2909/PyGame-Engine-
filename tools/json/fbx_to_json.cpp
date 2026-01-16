#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>

#define UFBX_IMPLEMENTATION
#include "../ufbx/ufbx.h"

// We use nlohmann/json single-header
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static std::array<float, 16> flatten_mat4(const ufbx_matrix &m)
{
    // ufbx_matrix is 3x4 (no last row), expand to 4x4
    // Column-major (OpenGL style)
    return {
        (float)m.m00, (float)m.m10, (float)m.m20, 0.0f,
        (float)m.m01, (float)m.m11, (float)m.m21, 0.0f,
        (float)m.m02, (float)m.m12, (float)m.m22, 0.0f,
        (float)m.m03, (float)m.m13, (float)m.m23, 1.0f,
    };
}

static void export_vec3_curve(json &out, const ufbx_anim_curve *curve)
{
    json times = json::array();
    json values = json::array();

    for (size_t i = 0; i < curve->keyframes.count; i++) {
        const ufbx_keyframe &k = curve->keyframes.data[i];
        times.push_back((double)k.time);
        values.push_back((double)k.value);
    }

    out["times"] = times;
    out["values"] = values;
}

static void export_quat_curve(json &out, const ufbx_anim_curve *curve)
{
    json times = json::array();
    json values = json::array();

    for (size_t i = 0; i < curve->keyframes.count; i++) {
        const ufbx_keyframe &k = curve->keyframes.data[i];
        times.push_back((double)k.time);
        values.push_back({
            (double)k.value
        });
    }

    out["times"] = times;
    out["values"] = values;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: fbx_to_json <file.fbx>\n");
        return 1;
    }

    const char *path = argv[1];

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(path, nullptr, &error);
    if (!scene) {
        fprintf(stderr, "ufbx load failed: %s\n", error.description.data);
        return 1;
    }

    json root;

    // ------------------------------------------------------------
    // Meta
    // ------------------------------------------------------------
    root["meta"] = {
        {"generator", "ufbx"},
        {"source", path},
        {"version", 1},
        {"unit_scale", 1.0}
    };

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

        // Inverse bind will be filled later from skin clusters if available
        ufbx_matrix inv_bind = ufbx_identity_matrix;

        bones.push_back({
            {"index", bone_index},
            {"name", std::string(node->name.data)},
            {"parent", parent},
            {"inverse_bind", flatten_mat4(inv_bind)} // identity for now
        });

        bone_index++;
    }

    root["skeleton"] = {
        {"bones", bones}
    };

    // ------------------------------------------------------------
    // Mesh + Skinning (first mesh only)
    // ------------------------------------------------------------
    if (scene->meshes.count > 0) {
        ufbx_mesh *mesh = scene->meshes.data[0];

        json joints = json::array();
        json weights = json::array();

        if (mesh->skin_deformers.count > 0) {
            ufbx_skin_deformer *skin = mesh->skin_deformers.data[0];

            for (size_t v = 0; v < mesh->num_vertices; v++) {
                std::array<int, 4> j = {0, 0, 0, 0};
                std::array<float, 4> w = {0.f, 0.f, 0.f, 0.f};

                int count = 0;

                for (size_t c = 0; c < skin->clusters.count && count < 4; c++) {
                    ufbx_skin_cluster *cluster = skin->clusters.data[c];
                    int bone = -1;
                    int eid = cluster->bone_node->element_id;
                    if (eid >= 0 && eid < (int)node_to_bone.size()) {
                        bone = node_to_bone[eid];
                    }

                    for (size_t wi = 0; wi < cluster->weights.count && count < 4; wi++) {
                        uint32_t vert = cluster->vertices.data[wi];
                        if ((size_t)vert == v) {
                            j[count] = bone >= 0 ? bone : 0;
                            w[count] = (float)cluster->weights.data[wi];
                            count++;
                        }
                    }
                }

                float sum = w[0] + w[1] + w[2] + w[3];
                if (sum > 0.0f) {
                    for (int k = 0; k < 4; k++) w[k] /= sum;
                }

                joints.push_back({j[0], j[1], j[2], j[3]});
                weights.push_back({w[0], w[1], w[2], w[3]});
            }
        }

        root["mesh"] = {
            {"vertex_count", (int)mesh->num_vertices},
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
        opts.resample_rate = 30.0; // export @ 30 fps

        for (size_t a = 0; a < scene->anim_stacks.count; a++) {
            ufbx_anim_stack *stack = scene->anim_stacks.data[a];

            // bake the animation
            ufbx_baked_anim *baked = ufbx_bake_anim(scene, stack->anim, &opts, &err);
            if (!baked) continue;

            json clip;
            clip["duration"] = baked->playback_duration;

            json channels = json::array();

            for (size_t i = 0; i < baked->nodes.count; i++) {
                ufbx_baked_node &bn = baked->nodes.data[i];

                // Resolve node from baked node element_id (ufbx >= 0.14)
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
                if (bone >= (int)bones.size()) continue;

                json channel;
                channel["bone"] = bone;

                // translation
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

                // rotation
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

                // scale
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

        root["animations"] = animations;
    }

    // ------------------------------------------------------------
    // Output
    // ------------------------------------------------------------
    printf("%s\n", root.dump(2).c_str());

    ufbx_free_scene(scene);
    return 0;
}