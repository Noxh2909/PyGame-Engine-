#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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

static void write_texture_png(const std::string &path, const ufbx_texture *tex)
{
    if (!tex || !tex->content.size) return;
    
    // Write the raw content to file (PNG/JPEG/etc)
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(tex->content.data, 1, tex->content.size, f);
    fclose(f);
}

static void write_json(const std::string &path, const json &j)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    std::string s = j.dump(2);
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
}

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
    std::string base = path;
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(path, nullptr, &error);
    if (!scene) {
        fprintf(stderr, "ufbx load failed: %s\n", error.description.data);
        return 1;
    }

    const ufbx_texture *albedo_tex = nullptr;
    for (size_t i = 0; i < scene->materials.count && !albedo_tex; i++) {
        ufbx_material *mat = scene->materials.data[i];

        if (mat->pbr.base_color.texture) {
            albedo_tex = mat->pbr.base_color.texture;
            break;
        }

        if (mat->fbx.diffuse_color.texture) {
            albedo_tex = mat->fbx.diffuse_color.texture;
            break;
        }
    }

    json mesh_json;
    json skeleton_json;
    json anim_json;

    // ------------------------------------------------------------
    // Skeleton (bind_pose only)
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
            if (pid >= 0 && pid < (int)node_to_bone.size())
                parent = node_to_bone[pid];
        }

        bones.push_back({
            {"name", std::string(node->name.data)},
            {"parent", parent},
            {"bind_pose", flatten_mat4(node->node_to_parent)}
        });

        bone_index++;
    }

    skeleton_json = {
        {"type", "skeleton"},
        {"bones", bones}
    };

    // ------------------------------------------------------------
    // Mesh (interleaved vertices + AABB)
    // ------------------------------------------------------------
    if (scene->meshes.count > 0) {
        ufbx_mesh *mesh = scene->meshes.data[0];

        // ------------------------------------------------------------
        // Compute smooth normals (angle-weighted per control point)
        // ------------------------------------------------------------
        std::vector<ufbx_vec3> smooth_normals(mesh->num_vertices);
        std::vector<float> smooth_counts(mesh->num_vertices, 0.0f);

        // Accumulate face normals
        for (size_t i = 0; i < mesh->num_indices; i += 3) {
            uint32_t i0 = mesh->vertex_indices.data[i + 0];
            uint32_t i1 = mesh->vertex_indices.data[i + 1];
            uint32_t i2 = mesh->vertex_indices.data[i + 2];

            ufbx_vec3 p0 = mesh->vertices.data[i0];
            ufbx_vec3 p1 = mesh->vertices.data[i1];
            ufbx_vec3 p2 = mesh->vertices.data[i2];

            ufbx_vec3 e1 = { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
            ufbx_vec3 e2 = { p2.x - p0.x, p2.y - p0.y, p2.z - p0.z };

            ufbx_vec3 fn = {
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x
            };

            smooth_normals[i0].x += fn.x;
            smooth_normals[i0].y += fn.y;
            smooth_normals[i0].z += fn.z;

            smooth_normals[i1].x += fn.x;
            smooth_normals[i1].y += fn.y;
            smooth_normals[i1].z += fn.z;

            smooth_normals[i2].x += fn.x;
            smooth_normals[i2].y += fn.y;
            smooth_normals[i2].z += fn.z;

            smooth_counts[i0] += 1.0f;
            smooth_counts[i1] += 1.0f;
            smooth_counts[i2] += 1.0f;
        }

        // Normalize accumulated normals
        for (size_t i = 0; i < mesh->num_vertices; i++) {
            ufbx_vec3 &n = smooth_normals[i];
            float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            if (len > 0.0f) {
                n.x /= len;
                n.y /= len;
                n.z /= len;
            }
        }

        json vertices = json::array();
        json indices  = json::array();

        float minx=1e30f, miny=1e30f, minz=1e30f;
        float maxx=-1e30f, maxy=-1e30f, maxz=-1e30f;

        for (size_t i = 0; i < mesh->num_indices; i++) {
            uint32_t cp = mesh->vertex_indices.data[i];

            auto &p = mesh->vertices.data[cp];
            ufbx_vec3 n = smooth_normals[cp];
            ufbx_vec2 uv = ufbx_get_vertex_vec2(&mesh->vertex_uv, i);

            std::array<int,4> j = {0,0,0,0};
            std::array<float,4> w = {0,0,0,0};
            int count = 0;

            if (mesh->skin_deformers.count > 0) {
                ufbx_skin_deformer *skin = mesh->skin_deformers.data[0];
                for (size_t c = 0; c < skin->clusters.count && count < 4; c++) {
                    ufbx_skin_cluster *cluster = skin->clusters.data[c];
                    int bone = node_to_bone[cluster->bone_node->element_id];

                    for (size_t wi = 0; wi < cluster->weights.count && count < 4; wi++) {
                        if (cluster->vertices.data[wi] == cp) {
                            j[count] = bone;
                            w[count] = (float)cluster->weights.data[wi];
                            count++;
                        }
                    }
                }
            }

            float sum = w[0]+w[1]+w[2]+w[3];
            if (sum > 0.0f) {
                for (int k = 0; k < 4; k++) w[k] /= sum;
            }

            int vtx_index = (int)vertices.size();

            vertices.push_back({
                p.x, p.y, p.z,
                n.x, n.y, n.z,
                uv.x, uv.y,
                j[0], j[1], j[2], j[3],
                w[0], w[1], w[2], w[3]
            });

            minx = std::min(minx, (float)p.x); miny = std::min(miny, (float)p.y); minz = std::min(minz, (float)p.z);
            maxx = std::max(maxx, (float)p.x); maxy = std::max(maxy, (float)p.y); maxz = std::max(maxz, (float)p.z);

            indices.push_back(vtx_index);
        }

        mesh_json = {
            {"type", "mesh"},
            {"vertex_layout", {"POSITION","NORMAL","UV","JOINTS","WEIGHTS"}},
            {"vertices", vertices},
            {"indices", indices},
            {"aabb", {
                {"min", {minx,miny,minz}},
                {"max", {maxx,maxy,maxz}}
            }}
        };
    }

    // ------------------------------------------------------------
    // Animation (baked tracks)
    // ------------------------------------------------------------
    json tracks = json::array();

    if (scene->anim_stacks.count > 0) {
        ufbx_bake_opts opts = {};
        opts.trim_start_time = true;
        opts.resample_rate = 30.0; // fixed-rate bake for runtime

        // Take first animation stack by default
        ufbx_anim_stack *stack = scene->anim_stacks.data[0];

        ufbx_error bake_err;
        ufbx_baked_anim *baked = ufbx_bake_anim(scene, stack->anim, &opts, &bake_err);
        if (baked) {

            for (size_t i = 0; i < baked->nodes.count; i++) {
                const ufbx_baked_node &bn = baked->nodes.data[i];

                if (bn.element_id >= scene->elements.count) continue;
                ufbx_element *elem = scene->elements.data[bn.element_id];
                if (!elem || elem->type != UFBX_ELEMENT_NODE) continue;

                ufbx_node *node = (ufbx_node*)elem;
                int bone = node_to_bone[node->element_id];
                if (bone < 0) continue;

                json track;
                track["bone"] = bone;

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
                    track["translation"] = t;
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
                    track["rotation"] = r;
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
                    track["scale"] = s;
                }

                if (track.size() > 1)
                    tracks.push_back(track);
            }

            anim_json = {
                {"type", "animation"},
                {"name", stack->name.length > 0 ? std::string(stack->name.data) : "anim"},
                {"duration", baked->playback_duration},
                {"tracks", tracks}
            };

            ufbx_free_baked_anim(baked);
        }
    }

    // ------------------------------------------------------------
    // Output files
    // ------------------------------------------------------------
    if (albedo_tex) {
        write_texture_png(base + ".albedo.png", albedo_tex);
    }
    write_json(base + ".mesh.json", mesh_json);
    write_json(base + ".skeleton.json", skeleton_json);
    if (!anim_json.is_null())
        write_json(base + ".anim.json", anim_json);

    ufbx_free_scene(scene);
    return 0;
}