#pragma once

#include <string>
#include <vector>

// Compiles MSL to a metallib via xcrun; Apple ships no library interface for it.
struct AirCompiler
{
    // Returns the metallib bytes, or empty with `error` set on failure.
    static std::vector<uint8_t> compile(const std::string& shaderSource, std::string& error);
};
