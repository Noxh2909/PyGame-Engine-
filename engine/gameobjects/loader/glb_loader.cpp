#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

#include "json_util.hpp"
#include "json.hpp"

static std::array<float, 16> flatten_matrix(const aiMatrix4x4& m) {
    return {
        m.a1, m.b1, m.c1, m.d1,
        m.a2, m.b2, m.c2, m.d2,
        m.a3, m.b3, m.c3, m.d3,
        m.a4, m.b4, m.c4, m.d4
    };
}

struct BoneInfo {
    std::string name;
    int parent;                 // parent bone index (-1 if root)
    aiMatrix4x4 inverse_bind;   // offset matrix
};

class GLBLoader {
public:
    bool load(const std::string& path) {
        scene = importer.ReadFile(
            path,
            aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_LimitBoneWeights
        );

        if (!scene || !scene->HasMeshes()) {
            std::cerr << "Failed to load GLB: " << path << std::endl;
            return false;
        }

        std::cout << "Loaded GLB: " << path << std::endl;
        std::cout << "Meshes: " << scene->mNumMeshes << std::endl;
        std::cout << "Animations: " << scene->mNumAnimations << std::endl;

        extract_skeleton(scene->mMeshes[0]);
        return true;
    }

    const std::vector<BoneInfo>& get_bones() const {
        return bones;
    }

    JsonAsset build_json_asset(const std::string& source_path) const {
        JsonAsset asset;

        // ---- meta ----
        asset.meta.generator = "glb/fbx_loader";
        asset.meta.version = 1;
        asset.meta.unit_scale = 1.0f;
        asset.meta.source = source_path;

        // ---- skeleton ----
        for (size_t i = 0; i < bones.size(); ++i) {
            const BoneInfo& b = bones[i];

            JsonBone jb;
            jb.index = static_cast<int>(i);
            jb.name = b.name;
            jb.parent = b.parent;
            jb.inverse_bind = flatten_matrix(b.inverse_bind);

            asset.skeleton.bones.push_back(jb);
        }

        return asset;
    }

private:
    Assimp::Importer importer;   // MUST live as long as scene
    const aiScene* scene = nullptr;

    std::vector<BoneInfo> bones;
    std::unordered_map<std::string, int> bone_index_by_name;

    void extract_skeleton(aiMesh* mesh) {
        std::cout << "Extracting skeleton..." << std::endl;

        // 1) Create bones from mesh
        for (unsigned i = 0; i < mesh->mNumBones; ++i) {
            aiBone* b = mesh->mBones[i];

            BoneInfo info;
            info.name = b->mName.C_Str();
            info.parent = -1; // assigned later
            info.inverse_bind = b->mOffsetMatrix;

            int index = static_cast<int>(bones.size());
            bones.push_back(info);
            bone_index_by_name[info.name] = index;

            std::cout << "Bone " << index << ": " << info.name << std::endl;
        }

        // 2) Resolve hierarchy via aiNode tree
        resolve_parents(scene->mRootNode, -1);
    }

    void resolve_parents(aiNode* node, int parent_bone) {
        std::string node_name = node->mName.C_Str();

        int current_bone = parent_bone;
        auto it = bone_index_by_name.find(node_name);
        if (it != bone_index_by_name.end()) {
            current_bone = it->second;
            bones[current_bone].parent = parent_bone;
        }

        for (unsigned i = 0; i < node->mNumChildren; ++i) {
            resolve_parents(node->mChildren[i], current_bone);
        }
    }
};

// Temporary test entry point (remove later)
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: glb_loader <file.glb/fbx>" << std::endl;
        return 1;
    }

    GLBLoader loader;
    if (!loader.load(argv[1])) {
        return 1;
    }

    const auto& bones = loader.get_bones();

    JsonAsset asset = loader.build_json_asset(argv[1]);

    std::string out_path = std::string(argv[1]) + ".json";
    write_json_asset(asset, out_path);

    std::cout << "Wrote JSON asset: " << out_path << std::endl;

    return 0;
}
