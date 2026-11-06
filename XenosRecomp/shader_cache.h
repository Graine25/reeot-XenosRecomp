#pragma once

#include <cstddef>
#include <cstdint>

namespace rengii::gpu
{
enum class ShaderStage : uint32_t { Vertex, Pixel };

struct ShaderVertexFetch
{
    uint32_t instruction; // Address in 12-byte instruction units.
    uint32_t mini;
};

struct ShaderCacheEntry
{
    uint64_t hash; // XXH3 of the complete, unmodified container.
    uint32_t spirvOffset; // Offset and size in the decompressed SMOL-V stream.
    uint32_t spirvSize;
    ShaderStage stage;
    uint32_t specConstantsMask;
    uint32_t vertexFetchOffset; // Index into g_shaderVertexFetches.
    uint32_t vertexFetchCount;
    uint32_t physicalCodeOffset;
    uint32_t floatConstantRegs;
    uint32_t interpolantMask;
    uint32_t textureSlotMask;
    uint32_t pointSlotMask;
    uint32_t linearSlotMask;
    uint32_t mipPointSlotMask;
    uint32_t mipLinearSlotMask;
    uint32_t mipBaseSlotMask;
    uint32_t anisoOffSlotMask;
    bool usesMemExport;
    bool coverageOutput; // Location 0, index 1; requires dual-source blending.
};

extern const ShaderCacheEntry g_shaderCacheEntries[];
extern const size_t g_shaderCacheEntryCount;
extern const ShaderVertexFetch g_shaderVertexFetches[];
extern const size_t g_shaderVertexFetchCount;
extern const uint8_t g_compressedSpirvCache[];
extern const size_t g_spirvCacheCompressedSize;
extern const size_t g_spirvCacheDecompressedSize;
}
