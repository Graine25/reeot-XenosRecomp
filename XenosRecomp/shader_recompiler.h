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

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
// Per-vertex-element vfetch instruction map captured during recompile. EOT ships
// containers with PLACEHOLDER vfetch fields (slot/offset/stride/format all zero)
// that the runtime patches per mesh decl (PatchVertexShaderToMatchVertexDecl-
// style), so the true layout only exists in the LIVE patched microcode. The
// cache therefore records WHERE each semantic's vfetch instruction lives; the
// reeot runtime decodes the patched fields from guest memory at draw time.
//   w0: usage:4 | usageIndex:4 | isMiniFetch:1
//   w1: instrAddress:16 | parentFullFetchAddress:16  (96-bit instruction units,
//       relative to the physical code start = ShaderCacheEntry.vfetch_code_offset)
struct VertexFetchLayoutRecord
{
    uint32_t w0;
    uint32_t w1;
};
#endif

struct ShaderRecompiler : StringBuffer
{
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
    // Structured if/else emission: instruction indices where a "} else {" is
    // emitted (then-branch close is part of the event, not ifEndLabels).
    std::unordered_set<uint32_t> elseLabels;
    uint32_t specConstantsMask = 0;

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
    std::vector<VertexFetchLayoutRecord> vertexLayout;
    uint32_t lastFullVfetchAddress = 0;  // mini-fetches inherit the previous full fetch's constant/stride
    uint32_t physicalCodeOffset = 0;     // instruction base within the physical microcode part
    // True when the shader's constant table has Float4 constants (a VS without
    // any has no WVP -> it outputs window-space positions; the reeot runtime
    // needs this to pick pixel->NDC conversion, and can't reflect it from the
    // DXIL because spec-constant shaders are stored as libraries).
    bool usesFloatConstants = false;
#endif

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

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
