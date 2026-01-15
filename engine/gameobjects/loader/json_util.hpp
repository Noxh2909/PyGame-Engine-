#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

// ============================================================
// JSON Asset Structures
// These mirror the exported anim_cache JSON exactly.
// ============================================================

// ---------------- Meta ----------------

struct JsonMeta {
    std::string generator;
    int version;
    float unit_scale;
    std::string source;
};

// ---------------- Skeleton ----------------

struct JsonBone {
    int index;
    std::string name;
    int parent; // -1 if root
    std::array<float, 16> inverse_bind;
};

struct JsonSkeleton {
    std::vector<JsonBone> bones;
};

// ---------------- Mesh Skinning ----------------

struct JsonMeshSkinning {
    int vertex_count = 0;

    // Per-vertex joint indices (max 4)
    std::vector<std::array<int, 4>> joints;

    // Per-vertex weights (max 4)
    std::vector<std::array<float, 4>> weights;
};

// ---------------- Animation ----------------

struct JsonAnimChannel {
    int bone; // bone index

    std::vector<float> pos_times;
    std::vector<std::array<float, 3>> pos_values;

    std::vector<float> rot_times;
    std::vector<std::array<float, 4>> rot_values;

    std::vector<float> scale_times;
    std::vector<std::array<float, 3>> scale_values;
};

struct JsonAnimation {
    std::string name;
    float duration = 0.0f;
    float ticks_per_second = 0.0f;
    std::vector<JsonAnimChannel> channels;
};

// ---------------- Top-Level Asset ----------------

struct JsonAsset {
    JsonMeta meta;
    JsonSkeleton skeleton;
    JsonMeshSkinning mesh;

    // animations indexed by name (e.g. "idle", "walk")
    std::unordered_map<std::string, JsonAnimation> animations;
};

// ============================================================
// Loader / Writer API
// Implementations live in json_util.cpp
// ============================================================

// Load asset from JSON file
JsonAsset load_json_asset(const std::string& path);

// Write asset to JSON file
void write_json_asset(const JsonAsset& asset, const std::string& path);
