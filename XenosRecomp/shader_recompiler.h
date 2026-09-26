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
    // The VS->PS interpolants, one bit per entry of INTERPOLATORS (TEXCOORD0-15
    // = bits 0-15, COLOR0-1 = 16-17, NORMAL0-1 = 18-19): the ones this shader's
    // container table says it writes (VS) or reads (PS). The signature declares
    // only those, so the host VS exports 4-6 attributes instead of 21 (the
    // driver does not prune exports the PS never reads).
    uint32_t interpolantMask = 0;
    // Vertex shaders only. Trimmed declares the mask; Full declares every
    // interpolant (for a PS reading one the VS never writes: the console left
    // it zero, and so does this); PositionOnly declares none, for draws
    // without a pixel shader (shadow casters, depth priming), where the
    // compiler then drops the interpolant maths as well. Velocity (VS and
    // PS) is the motion-vector pair: the VS runs its body twice, the first
    // pass reading a previous-frame copy of its float constant file placed
    // 4 KB (c256) after the current one in the same buffer, and exports the
    // two clip positions on VELOCITY0/1; the PS declares those inputs and
    // writes their screen-space difference to SV_Target1.
    enum class InterpolantVariant : uint8_t { Trimmed, Full, PositionOnly, Velocity };
    InterpolantVariant interpolantVariant = InterpolantVariant::Trimmed;
    bool velocityVertexShader() const { return !isPixelShader && interpolantVariant == InterpolantVariant::Velocity; }
    // Vertex shaders: every ALU instruction in program order -- the temps its
    // vector and scalar halves write and read, whether its scalar half
    // computes ps (and reads the previous ps), for a dp3/dp4 the offset of
    // the "dotF" printed for it, and the temps its position export reads.
    // The dots the position is computed from become dotP once the shader is
    // printed (shader_common.h).
    struct VertexAluWrite
    {
        uint32_t vectorDest;
        uint32_t vectorMask;
        uint64_t vectorSources;
        size_t dotOffset; // SIZE_MAX when not a dot
        uint32_t scalarDest;
        uint32_t scalarMask; // a temp written from ps
        bool scalarOp;       // computes ps
        bool scalarReadsPs;  // ...from the previous ps
        uint64_t scalarSources;
        bool conditional;    // predicated, or inside a branch or a loop
        uint64_t positionSources;
        bool positionReadsPs;
    };
    std::vector<VertexAluWrite> vertexAluWrites;
    // Where each temp's declaration was printed (0: not printed), and ps's, so
    // the ones the position is computed through can be made precise afterwards.
    size_t registerDeclOffsets[32]{};
    size_t psDeclOffset = 0;
    // False while printing a pc/switch program, where the order the
    // instructions are printed in is not the order they run in.
    bool straightLineFlow = true;
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
