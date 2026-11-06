// NGII paths adapted from YaibaKamikaze/IzunaRecomp, fork fb51ac62 (MIT).
#pragma once

#include "shader.h"
#include "shader_code.h"

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

// Instruction address in 12-byte units, followed by the mini-fetch flag.
struct VertexFetchLayoutRecord
{
    uint32_t w0;
    uint32_t w1;
};

struct ShaderRecompiler : StringBuffer
{
    std::string fetchIndex;
    uint64_t unsupportedExports = 0; // Bit per export register without a destination.
    std::unordered_set<uint32_t> literalConstants;
    bool usesMemExport = false;

    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElement> vertexElements;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> int4Constants;
    std::unordered_map<uint32_t, const char*> samplers;
    std::unordered_map<uint32_t, uint32_t> ifEndLabels;
    std::unordered_set<uint32_t> elseLabels;
    uint32_t specConstantsMask = 0;
    bool coverageOutput = false;

    std::vector<VertexFetchLayoutRecord> vertexLayout;
    uint32_t physicalCodeOffset = 0;
    bool usesFloatConstants = false;
    uint32_t interpolantMask = 0;
    uint32_t floatConstantRegs = 0; // Upload bound; relative reads need all 256 registers.
    uint32_t textureSlotMask = 0;
    // Requirements from fetch instructions; a slot can appear in conflicting masks.
    uint32_t pointSlotMask = 0;
    uint32_t linearSlotMask = 0;
    uint32_t mipPointSlotMask = 0;
    uint32_t mipLinearSlotMask = 0;
    uint32_t mipBaseSlotMask = 0;
    uint32_t anisoOffSlotMask = 0;
    enum class InterpolantVariant : uint8_t { Trimmed, Full, PositionOnly, Velocity };
    InterpolantVariant interpolantVariant = InterpolantVariant::Trimmed;
    bool velocityVertexShader() const { return !isPixelShader && interpolantVariant == InterpolantVariant::Velocity; }
    struct VertexAluWrite
    {
        uint32_t vectorDest;
        uint32_t vectorMask;
        uint64_t vectorSources;
        size_t dotOffset;
        uint32_t scalarDest;
        uint32_t scalarMask;
        bool scalarOp;
        bool scalarReadsPs;
        uint64_t scalarSources;
        bool conditional;
        uint64_t positionSources;
        bool positionReadsPs;
    };
    std::vector<VertexAluWrite> vertexAluWrites;
    size_t registerDeclOffsets[32]{};
    size_t psDeclOffset = 0;
    bool straightLineFlow = true;

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    void recompile(const uint8_t* shaderData, const std::string_view& include);
};
