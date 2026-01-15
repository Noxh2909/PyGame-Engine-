

#include "json_util.hpp"
#include "json.hpp"

#include <fstream>
#include <stdexcept>

// ============================================================
// Write JsonAsset to disk
// ============================================================

void write_json_asset(const JsonAsset& asset, const std::string& path)
{
    json root;

    // ---------------- meta ----------------
    root["meta"] = {
        {"generator", asset.meta.generator},
        {"version", asset.meta.version},
        {"unit_scale", asset.meta.unit_scale},
        {"source", asset.meta.source}
    };

    // ---------------- skeleton ----------------
    json bones_json = json::array();

    for (const JsonBone& b : asset.skeleton.bones) {
        json bone_json;
        bone_json["index"] = b.index;
        bone_json["name"] = b.name;
        bone_json["parent"] = b.parent;

        // inverse bind matrix (flattened 4x4)
        json inv_bind = json::array();
        for (float v : b.inverse_bind) {
            inv_bind.push_back(v);
        }
        bone_json["inverse_bind"] = inv_bind;

        bones_json.push_back(bone_json);
    }

    root["skeleton"] = {
        {"bones", bones_json}
    };

    // ---------------- write file ----------------
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open JSON output file: " + path);
    }

    // pretty print with indentation = 2
    out << root.dump(2);
    out.close();
}

// ============================================================
// JSON loading (not implemented yet)
// ============================================================

JsonAsset load_json_asset(const std::string& /*path*/)
{
    throw std::runtime_error("load_json_asset() not implemented yet");
}