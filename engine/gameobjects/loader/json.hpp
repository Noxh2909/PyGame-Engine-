#pragma once

// ============================================================
// json.hpp (project-local wrapper)
// ============================================================
//
// This project uses the nlohmann/json library.
//
// REQUIRED SETUP (choose ONE):
//
// Option A (RECOMMENDED, project-local):
//   - Download the single-header file `json.hpp` from:
//     https://github.com/nlohmann/json/releases
//   - Place it in THIS directory, replacing this file.
//
// Option B (system-wide, via Homebrew):
//   brew install nlohmann-json
//   and change includes to:
//     #include <nlohmann/json.hpp>
//
// ------------------------------------------------------------
// Why this wrapper exists:
// - Keeps glb_loader / tools self-contained
// - Avoids hidden system dependencies
// - Makes future vendoring or replacement trivial
// ============================================================

#include <nlohmann/json.hpp>

// Project-wide alias
using json = nlohmann::json;
