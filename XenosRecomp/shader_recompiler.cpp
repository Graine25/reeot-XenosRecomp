// NGII paths adapted from YaibaKamikaze/IzunaRecomp, fork fb51ac62 (MIT).
#include "shader_recompiler.h"
#include "shader_common.h"

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
    "float4", // NORMAL
    "float4", // PSIZE
    "float4", // TEXCOORD
    "float4", // TANGENT
    "float4", // BINORMAL
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 },
    { DeclUsage::Normal, 0 },
    { DeclUsage::Normal, 1 }
};

static constexpr std::string_view TEXTURE_DIMENSIONS[] =
{
    "2D",
    "3D",
    "Cube",
    "1D"
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    // D3D patches these fields when it binds the declaration, so decode them at draw time.
    // Mini fetches reuse the last full fetch's index, slot and stride.
    if (!instr.isMiniFetch)
    {
        static constexpr char components[] = "xyzw";
        fetchIndex = fmt::format("r{}.{}", instr.srcRegister, components[instr.srcSwizzle & 3]);
    }
    const uint32_t k = uint32_t(vertexLayout.size());
    if (k >= 64)
        throw std::runtime_error("more than 64 vfetches");
    vertexLayout.push_back({address, instr.isMiniFetch ? 1u : 0u});
    print("rengiiVertexFetchK({}, {})", k, fetchIndex.empty() ? std::string("0.0") : fetchIndex);
    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);
    out += ";\n";
    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    // [rengii] setTexLOD / setGradientH / setGradientV: state for later fetches
    // (shader_common.h, rengiiLodMode).
    if (instr.opcode == FetchOpcode::SetTextureLod || instr.opcode == FetchOpcode::SetTextureGradientsHorz ||
        instr.opcode == FetchOpcode::SetTextureGradientsVert)
    {
        if (instr.isPredicated)
        {
            indent();
            println("if ({}p0)", instr.predCondition ? "" : "!");
        }
        indent();
        if (instr.opcode == FetchOpcode::SetTextureLod)
            println("rengiiRegLod = r{}.{};", instr.srcRegister, SWIZZLES[instr.srcSwizzle & 0x3]);
        else
            println("{} = r{}.{}{}{};", instr.opcode == FetchOpcode::SetTextureGradientsHorz ? "rengiiGradH" : "rengiiGradV",
                    instr.srcRegister, SWIZZLES[instr.srcSwizzle & 0x3], SWIZZLES[(instr.srcSwizzle >> 2) & 0x3],
                    SWIZZLES[(instr.srcSwizzle >> 4) & 0x3]);
        return;
    }
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights)
        return;

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    std::string constName;
    const char* constNamePtr = nullptr;

    textureSlotMask |= 1u << (instr.constIndex & 31);
    if (instr.magFilter == 0 || instr.minFilter == 0)
        pointSlotMask |= 1u << (instr.constIndex & 31);
    else if (instr.magFilter == 1 || instr.minFilter == 1)
        linearSlotMask |= 1u << (instr.constIndex & 31);
    if (instr.opcode == FetchOpcode::TextureFetch)
    {
        if (instr.mipFilter == 0) mipPointSlotMask |= 1u << (instr.constIndex & 31);
        else if (instr.mipFilter == 1) mipLinearSlotMask |= 1u << (instr.constIndex & 31);
        else if (instr.mipFilter == 2) mipBaseSlotMask |= 1u << (instr.constIndex & 31);
        if (instr.anisoFilter == 0) anisoOffSlotMask |= 1u << (instr.constIndex & 31);
    }
    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

    // [rengii] The fetch's mip selection (rengiiLodMode): UseComputedLOD (only in
    // pixel shaders, or with register gradients), the register LOD and the
    // instruction's LOD bias (signed 3.4 fixed point).
    if (instr.opcode == FetchOpcode::TextureFetch)
    {
        const float bias = float(instr.lodBias) * (1.0f / 16.0f);
        const bool computed = instr.useCompLod && (isPixelShader || instr.useRegGradients);
        const uint32_t mode = !computed ? 2u : instr.useRegGradients ? 3u : (instr.useRegLod || bias != 0.0f) ? 1u : 0u;
        indent();
        if (instr.useRegLod)
            println("rengiiLodMode = {}; rengiiLod = rengiiRegLod + {};", mode, bias);
        else
            println("rengiiLodMode = {}; rengiiLod = {};", mode, bias);
    }
    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {

        out += "applyFetchSign(tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

    print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex, ", constNamePtr, dimension);
    printSrcRegister(componentCount);

    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        // [rengii] (in guest texels: x S on a scaled resolve, rengiiOffsetScale)
        if (instr.offsetX != 0 || instr.offsetY != 0)
            print(", float2({}, {}) * rengiiOffsetScale({})", instr.offsetX * 0.5f, instr.offsetY * 0.5f, instr.constIndex);
        else
            out += ", float2(0, 0)";
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    if (instr.opcode == FetchOpcode::TextureFetch)
        print("), g_BiasedTextures, g_SintTexcoords, {}).", instr.constIndex);
    else
        out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

static uint64_t vectorSourceTemps(const AluInstruction& instr)
{
    uint32_t operands;
    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::Frc:
    case AluVectorOpcode::Trunc:
    case AluVectorOpcode::Floor:
    case AluVectorOpcode::Max4:
        operands = 1;
        break;
    case AluVectorOpcode::Mad:
    case AluVectorOpcode::CndEq:
    case AluVectorOpcode::CndGe:
    case AluVectorOpcode::CndGt:
    case AluVectorOpcode::Dp2Add:
        operands = 3;
        break;
    default:
        operands = 2;
        break;
    }

    uint64_t temps = 0;
    if (instr.src1Select)
        temps |= 1ull << (instr.src1Register & 0x3F);
    if (operands >= 2 && instr.src2Select)
        temps |= 1ull << (instr.src2Register & 0x3F);
    if (operands >= 3 && instr.src3Select)
        temps |= 1ull << (instr.src3Register & 0x3F);
    return temps;
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    VertexAluWrite vertexWrite{};
    vertexWrite.dotOffset = SIZE_MAX;
    vertexWrite.conditional = instr.isPredicated || !straightLineFlow || indentation > 1;
    bool vertexPositionExport = false;

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent(); 
        out += "{\n";
        ++indentation;
    }

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    bool scalarReadsVectorDest = false;
    if (!instr.exportData && instr.vectorWriteMask != 0 && instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        const bool constantScalar =
            instr.scalarOpcode == AluScalarOpcode::Mulsc0 || instr.scalarOpcode == AluScalarOpcode::Mulsc1 ||
            instr.scalarOpcode == AluScalarOpcode::Addsc0 || instr.scalarOpcode == AluScalarOpcode::Addsc1 ||
            instr.scalarOpcode == AluScalarOpcode::Subsc0 || instr.scalarOpcode == AluScalarOpcode::Subsc1;
        uint32_t scalarReg;
        bool scalarSelect;
        if (constantScalar)
        {
            scalarReg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
            scalarSelect = true;
        }
        else
        {
            scalarReg = instr.src3Register & 0x3F;
            scalarSelect = instr.src3Select;
        }
        scalarReadsVectorDest = scalarSelect && scalarReg == instr.vectorDest;
    }

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                const bool scalarOperand = operand == SCALAR_0 || operand == SCALAR_1 || operand == SCALAR_CONSTANT_1;
                if (scalarOperand && scalarReadsVectorDest && reg == instr.vectorDest)
                    regFormatted = "vpre";
                else
                    regFormatted = fmt::format("r{}", reg);
            }
            else
            {
                // (relative reads may reach any register: the whole file -
                // the emission below decides relative by const0Relative alone)
                // (a register the shader defines as a literal, not named by the
                // table, is a local - see below - and not read from the buffer)
                if (instr.const0Relative)
                    floatConstantRegs = 256u;
                else if (!(literalConstants.count(reg) && !float4Constants.count(reg)))
                    floatConstantRegs = std::max<uint32_t>(floatConstantRegs, uint32_t(reg) + 1);
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                            if (velocityVertexShader())
                                regFormatted = fmt::format("(velIter == 0 ? prev_{} : {})", regFormatted, regFormatted);
                        }
                    }
                    else
                    {
                        if (instr.const0Relative)
                        {
                            // [rengii] A single named constant indexed relatively: the
                            // name is one register, the read is not - by register.
                            regFormatted = fmt::format("vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + clamp({}{}, 0, 255) * 16, 0x10)",
                                isPixelShader ? "Pixel" : "Vertex", reg, instr.constAddressRegisterRelative ? " + a0" : " + aL");
                        }
                        else
                            regFormatted = constantName;
                        if (velocityVertexShader())
                            regFormatted = fmt::format("(velIter == 0 ? prev_{} : {})", constantName, constantName);
                    }
                }
                else
                {
                    // [rengii] A constant the table doesn't name (or no table):
                    // read the register straight from the stage's constants,
                    // relative to a0 / aL when the instruction says so.
                    // A register the shader defines as a literal is its local
                    // (c<N>), as the hardware takes the shader's definitions.
                    const char* relative = instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "";
                    if (!instr.const0Relative && literalConstants.count(reg))
                        regFormatted = fmt::format("c{}", reg);
                    else
                        regFormatted = fmt::format("vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + clamp({}{}, 0, 255) * 16, 0x10)",
                            isPixelShader ? "Pixel" : "Vertex", reg, relative);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

                vertexPositionExport = true;

                break;

            default:
            {
                auto findResult = interpolators.find(instr.vectorDest);
                if (instr.vectorDest >= 32 && instr.vectorDest <= 37)
                {
                    // [rengii] Memory export: eA (32) and eM0-eM4 (33-37) are
                    // locals; each eM write is stored (rengiiMemExport, below).
                    static constexpr const char* kMemExport[] = { "rengiieA", "rengiieM0", "rengiieM1", "rengiieM2", "rengiieM3", "rengiieM4" };
                    exportRegister = kMemExport[instr.vectorDest - 32];
                    break;
                }
                if (findResult == interpolators.end())
                {
                    // [rengii] An export this translator has no destination for
                    // (point size / kill vertex 63, memexport 32-37, ...):
                    // noted, written to a scratch value instead.
                    unsupportedExports |= 1ull << (instr.vectorDest & 63);
                    exportRegister = "rengiiUnsupportedExport";
                    break;
                }
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    if (scalarReadsVectorDest)
    {
        indent();
        println("vpre = r{};", instr.vectorDest);
    }

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            print("max({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            print("min({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            if (!isPixelShader)
            {
                if (exportRegister == "oPos")
                {
                    print("dotP({}, {})", op(VECTOR_0), op(VECTOR_1));
                }
                else
                {
                    if (!instr.exportData)
                        vertexWrite.dotOffset = out.size();
                    print("dotF({}, {})", op(VECTOR_0), op(VECTOR_1));
                }
                break;
            }
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            print("rengiiCube({})", op(VECTOR_0));  // [rengii] the hardware's result (shader_common.h)
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";

        if (!instr.exportData)
        {
            vertexWrite.vectorDest = instr.vectorDest & 0x3F;
            vertexWrite.vectorMask = vectorWriteMask;
            vertexWrite.vectorSources = vectorSourceTemps(instr);
        }
        else if (vertexPositionExport)
        {
            vertexWrite.positionSources = vectorSourceTemps(instr);
        }
    }

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            print("max({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            print("min({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(rcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            // PRED_SETINV: src==1 -> 0, src==0 -> 1, else passthrough.
            print("{0} == 1.0 ? 0.0 : ({0} == 0.0 ? 1.0 : {0})", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        vertexWrite.scalarOp = true;
        vertexWrite.scalarReadsPs =
            instr.scalarOpcode == AluScalarOpcode::AddsPrev || instr.scalarOpcode == AluScalarOpcode::SubsPrev ||
            instr.scalarOpcode == AluScalarOpcode::MulsPrev || instr.scalarOpcode == AluScalarOpcode::MulsPrev2;
        const bool constantScalar =
            instr.scalarOpcode == AluScalarOpcode::Mulsc0 || instr.scalarOpcode == AluScalarOpcode::Mulsc1 ||
            instr.scalarOpcode == AluScalarOpcode::Addsc0 || instr.scalarOpcode == AluScalarOpcode::Addsc1 ||
            instr.scalarOpcode == AluScalarOpcode::Subsc0 || instr.scalarOpcode == AluScalarOpcode::Subsc1;
        if (constantScalar)
            vertexWrite.scalarSources = 1ull << (((uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C)) & 0x3F);
        else if (instr.src3Select)
            vertexWrite.scalarSources = 1ull << (instr.src3Register & 0x3F);
    }
    if (scalarWriteMask != 0)
    {
        if (!instr.exportData)
        {
            vertexWrite.scalarDest = instr.scalarDest & 0x3F;
            vertexWrite.scalarMask = scalarWriteMask;
        }
        else if (vertexPositionExport)
        {
            vertexWrite.positionReadsPs = true;
        }
    }

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.exportData && !isPixelShader && instr.vectorDest >= 33 && instr.vectorDest <= 37)
    {
        indent();  // [rengii] store this eM element now (shader_common.h)
        println("rengiiMemExport(rengiieA, {0}, rengiieM{0});", instr.vectorDest - 33);
        usesMemExport = true;
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (!isPixelShader)
        vertexAluWrites.push_back(vertexWrite);
}

#include <cctype>
#include <cstring>
#include <string_view>
#include <vector>

// Snap only input components that reach comparisons; this keeps rigid limb weights exact.
// Complex control flow keeps every snap.
namespace
{
    constexpr uint32_t kRengiiSlots = 36;  // r0-r31, ps, p0, a0, aL

    // The input components (bit reg * 4 + component) a value carries.
    struct RengiiBits
    {
        uint64_t lo = 0, hi = 0;
        void set(uint32_t bit) { (bit < 64 ? lo : hi) |= 1ull << (bit & 63); }
        bool test(uint32_t bit) const { return ((bit < 64 ? lo : hi) >> (bit & 63)) & 1; }
        RengiiBits& operator|=(const RengiiBits& o) { lo |= o.lo; hi |= o.hi; return *this; }
    };

    int rengiiSlot(std::string_view name)
    {
        if (name == "ps") return 32;
        if (name == "p0") return 33;
        if (name == "a0") return 34;
        if (name == "aL") return 35;
        if (name.size() >= 2 && name[0] == 'r')
        {
            int n = 0;
            for (size_t i = 1; i < name.size(); ++i)
            {
                if (name[i] < '0' || name[i] > '9') return -1;
                n = n * 10 + (name[i] - '0');
            }
            return n < 32 ? n : -1;
        }
        return -1;
    }

    bool rengiiIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

    int rengiiComp(char c) { return c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : c == 'w' ? 3 : -1; }

    // What the registers named in `e` carry.
    RengiiBits rengiiRefTaint(std::string_view e, const RengiiBits (&taint)[kRengiiSlots][4])
    {
        RengiiBits out;
        for (size_t i = 0; i < e.size();)
        {
            if (!rengiiIdentChar(e[i]) || (i > 0 && rengiiIdentChar(e[i - 1])))
            {
                ++i;
                continue;
            }
            size_t j = i;
            while (j < e.size() && rengiiIdentChar(e[j])) ++j;
            const int slot = rengiiSlot(e.substr(i, j - i));
            if (slot >= 0)
            {
                std::string_view comps = slot >= 32 ? "x" : "xyzw";
                if (j + 1 < e.size() && e[j] == '.' && rengiiComp(e[j + 1]) >= 0)
                {
                    size_t k = j + 1;
                    while (k < e.size() && rengiiComp(e[k]) >= 0) ++k;
                    comps = e.substr(j + 1, k - j - 1);
                    j = k;
                }
                for (char c : comps)
                    if (const int n = rengiiComp(c); n >= 0) out |= taint[slot][n];
            }
            i = j;
        }
        return out;
    }

    bool rengiiHasCompare(std::string_view e)
    {
        return e.find('<') != std::string_view::npos || e.find('>') != std::string_view::npos ||
               e.find("==") != std::string_view::npos || e.find("!=") != std::string_view::npos;
    }

    // The parts of an expression that decide a comparison: a select's
    // condition, a ternary's condition, or the whole of a comparison.
    void rengiiConditionParts(std::string_view e, std::vector<std::string_view>& parts)
    {
        bool selected = false;
        for (size_t at = e.find("select("); at != std::string_view::npos; at = e.find("select(", at + 1))
        {
            selected = true;
            size_t i = at + 7;
            int depth = 0;
            const size_t start = i;
            for (; i < e.size(); ++i)
            {
                if (e[i] == '(' || e[i] == '[') ++depth;
                else if (e[i] == ')' || e[i] == ']') { if (depth-- == 0) break; }
                else if (e[i] == ',' && depth == 0) break;
            }
            parts.push_back(e.substr(start, i - start));
        }
        if (const size_t q = e.find('?'); q != std::string_view::npos)
            parts.push_back(e.substr(0, q));
        else if (!selected && rengiiHasCompare(e))
            parts.push_back(e);
    }

    std::string_view rengiiTrim(std::string_view s)
    {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
        return s;
    }
}

void rengiiNarrowInterpolantSnaps(std::string& out, bool simpleControlFlow)
{
    static constexpr std::string_view kCall = "rengiiInterpolated(iTexCoord";
    struct Input { size_t at; size_t lineEnd; int reg; };  // `at`: where "rengiiInterpolated(" begins
    std::vector<Input> inputs;
    size_t bodyStart = 0;
    for (size_t at = out.find(kCall); at != std::string::npos; at = out.find(kCall, at + 1))
    {
        const size_t lineStart = out.rfind('\n', at) + 1;
        const std::string_view line = std::string_view(out).substr(lineStart, at - lineStart);
        const size_t r = line.find("float4 r");
        if (r == std::string_view::npos) return;  // (not the form this pass knows: keep every snap)
        bodyStart = out.find('\n', at);
        if (bodyStart == std::string::npos) return;
        inputs.push_back({at, bodyStart, std::atoi(line.data() + r + 8)});
    }
    if (inputs.empty() || !simpleControlFlow) return;
    size_t bodyEnd = out.find("[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)", bodyStart);
    if (bodyEnd == std::string::npos) bodyEnd = out.size();
    const std::string_view body = std::string_view(out).substr(bodyStart, bodyEnd - bodyStart);
    if (body.find("while") != std::string_view::npos || body.find("for (") != std::string_view::npos ||
        body.find("[loop]") != std::string_view::npos || body.find("switch") != std::string_view::npos)
        return;

    RengiiBits taint[kRengiiSlots][4] = {};
    for (const Input& in : inputs)
        if (in.reg >= 0 && in.reg < 32)
            for (int c = 0; c < 4; ++c) taint[in.reg][c].set(uint32_t(in.reg * 4 + c));
    RengiiBits need;
    int depth = 0;
    std::vector<std::string_view> parts;
    for (size_t pos = 0; pos < body.size();)
    {
        size_t end = body.find('\n', pos);
        if (end == std::string_view::npos) end = body.size();
        const std::string_view s = rengiiTrim(body.substr(pos, end - pos));
        pos = end + 1;
        if (s.empty()) continue;
        if (const size_t ifAt = s.find("if ("); ifAt != std::string_view::npos && s.back() == ')')
            need |= rengiiRefTaint(s.substr(ifAt + 3), taint);
        for (char c : s) depth += c == '{' ? 1 : c == '}' ? -1 : 0;
        if (s.rfind("clip(", 0) == 0 || s.find("discard") != std::string_view::npos)
        {
            parts.clear();
            rengiiConditionParts(s.substr(s.find('(') + 1), parts);
            if (parts.empty()) need |= rengiiRefTaint(s, taint);
            for (std::string_view p : parts) need |= rengiiRefTaint(p, taint);
            continue;
        }
        // An assignment: `dst[.mask] = rhs;`
        size_t n = 0;
        while (n < s.size() && rengiiIdentChar(s[n])) ++n;
        const int slot = rengiiSlot(s.substr(0, n));
        if (slot < 0) continue;
        std::string_view mask = slot >= 32 ? "x" : "xyzw";
        size_t m = n;
        if (m < s.size() && s[m] == '.')
        {
            size_t k = m + 1;
            while (k < s.size() && rengiiComp(s[k]) >= 0) ++k;
            mask = s.substr(m + 1, k - m - 1);
            m = k;
        }
        while (m < s.size() && s[m] == ' ') ++m;
        if (m + 1 >= s.size() || s[m] != '=' || s[m + 1] == '=') continue;
        std::string_view rhs = s.substr(m + 1);
        if (!rhs.empty() && rhs.back() == ';') rhs.remove_suffix(1);
        parts.clear();
        rengiiConditionParts(rhs, parts);
        for (std::string_view p : parts) need |= rengiiRefTaint(p, taint);
        // (a texture's result is a new value: its coordinates' last bits are not what this is about)
        const RengiiBits src = rhs.find("tfetch") != std::string_view::npos ? RengiiBits{} : rengiiRefTaint(rhs, taint);
        for (char c : mask)
        {
            const int k = rengiiComp(c);
            if (k < 0) continue;
            if (depth <= 0) taint[slot][k] = src;
            else taint[slot][k] |= src;  // (inside a branch the old value may stand)
        }
    }

    // Back to front, so the offsets before each edit stay right. A line
    // "float4 rN = rengiiInterpolated(iTexCoordM);" becomes the plain read, then
    // (when some components are compared) a snap of just those.
    for (auto it = inputs.rbegin(); it != inputs.rend(); ++it)
    {
        if (it->reg < 0 || it->reg >= 32) continue;
        std::string comps;
        for (int c = 0; c < 4; ++c)
            if (need.test(uint32_t(it->reg * 4 + c))) comps += "xyzw"[c];
        if (comps.size() == 4) continue;
        if (!comps.empty())
        {
            const std::string r = "r" + std::to_string(it->reg) + "." + comps;
            out.insert(it->lineEnd, "\n\t" + r + " = rengiiInterpolated(" + r + ");");
        }
        const size_t close = out.find(')', it->at);
        out.erase(close, 1);                  // "...iTexCoordN)" -> "...iTexCoordN"
        out.erase(it->at, kCall.size() - 9);  // drop "rengiiInterpolated(" (keep "iTexCoord")
    }
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    assert((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100);

    // Vertex textures need explicit LOD sampling.
    out += fmt::format("#define RENGII_RECOMP 1\n#define RENGII_VERTEX_SHADER {}\n", (shaderContainer->flags & 0x1) ? 1 : 0);
    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    // [rengii] Some of Ninja Gaiden II's containers have no constant table
    // (offset 0): read an empty one, not the container header as a table.
    alignas(16) static const uint8_t emptyConstantTable[256] = {};
    const auto constantTableContainer = shaderContainer->constantTableOffset != NULL
        ? reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset)
        : reinterpret_cast<const ConstantTableContainer*>(emptyConstantTable);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    out += "#ifdef __spirv__\n\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                println("#define {}(INDEX) select((INDEX) < {}, vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
                if (velocityVertexShader())
                    println("#define prev_{}(INDEX) select((INDEX) < {}, vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + 4096 + ({} + min(INDEX, {})) * 16, 0x10), 0.0)",
                        constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10)",
                    constantName, shaderName, constantInfo->registerIndex * 16);
                if (velocityVertexShader())
                    println("#define prev_{} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + 4096 + {}, 0x10)",
                        constantName, shaderName, constantInfo->registerIndex * 16);
            }

            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            {
                uint32_t reg = constantInfo->registerIndex + j;
                auto it = float4Constants.find(reg);
                if (it == float4Constants.end() || it->second->registerCount < constantInfo->registerCount)
                    float4Constants[reg] = constantInfo;
            }

            break;
        }

        case RegisterSet::Sampler:
        {
            // [rengii] 32 fetch slots (NG2 samples through 16-31 too), in a region
            // of their own at SharedConstants + 1024: 128 bytes per dimension.
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, TEXTURE_DIMENSIONS[j], 1024 + j * 128 + constantInfo->registerIndex * 4);
            }
            println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                constantName, 1024 + std::size(TEXTURE_DIMENSIONS) * 128 + constantInfo->registerIndex * 4);

            samplers.emplace(constantInfo->registerIndex, constantName);
            break;
        }

        case RegisterSet::Int4:
        {
            const uint32_t loopRegister = uint32_t(constantInfo->registerIndex) + (isPixelShader ? 16u : 0u);
            int4Constants.emplace(loopRegister, constantName);
            break;
        }

        }
    }

    for (uint32_t r = 0; r < 32; r++)  // [rengii] all 32 fetch slots (see the named samplers)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("#define s{}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                r, TEXTURE_DIMENSIONS[j], 1024 + j * 128 + r * 4);
        }
        println("#define s{}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
            r, 1024 + std::size(TEXTURE_DIMENSIONS) * 128 + r * 4);
    }

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            usesFloatConstants = true;
            auto winner = float4Constants.find(constantInfo->registerIndex);
            if (winner == float4Constants.end() || winner->second != constantInfo)
                continue;

            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
            }
            if (velocityVertexShader())
            {
                print("\tfloat4 prev_{}", constantName);
                if (constantInfo->registerCount > 1)
                    print("[{}]", constantInfo->registerCount.get());
                println(" : packoffset(c{});", constantInfo->registerIndex.get() + 256);
                if (constantInfo->registerCount > 1)
                {
                    uint32_t tailCount = 256 - constantInfo->registerIndex;
                    println("#define prev_{0}(INDEX) select((INDEX) < {1}, prev_{0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
                }
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
        }
    }

    for (uint32_t r = 0; r < 16; r++)
    {
        if (samplers.find(r) != samplers.end())
            continue;
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
        {
            println("\tuint s{}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                r, TEXTURE_DIMENSIONS[j], j * 4 + r / 4, SWIZZLES[r % 4]);
        }
        println("\tuint s{}_SamplerDescriptorIndex : packoffset(c{}.{});",
            r, 4 * std::size(TEXTURE_DIMENSIONS) + r / 4, SWIZZLES[r % 4]);
    }

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            const uint32_t unifiedBit = uint32_t(constantInfo->registerIndex) + (isPixelShader ? 128u : 0u);
            println("\t#define {} BOOL_BIT({})", constantName, unifiedBit);
            boolConstants.emplace(unifiedBit, constantName);
        }
    }

    for (auto& [registerIndex, constantName] : int4Constants)
        println("\t#define {} int4(g_LoopConstants({}))", constantName, registerIndex);

    out += '\n';

    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);

    {
        const uint32_t count = (shader->interpolatorInfo >> 5) & 0x1F;
        for (uint32_t i = 0; i < count; i++)
        {
            union
            {
                Interpolator interpolator;
                uint32_t value;
            };
            if (isPixelShader)
            {
                value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            }
            else
            {
                auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
                value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            }
            // [rengii] Positional linkage: every interpolator is TEXCOORD<register>
            // (pixel: its input register; vertex: its export register), as the
            // Xenos hardware links them. NG2's containers often lack the tables
            // a semantic match would need; the microcode is what runs unchanged.
            interpolator.usage = DeclUsage::TexCoord;
            interpolator.usageIndex = isPixelShader ? uint32_t(interpolator.reg) : i;
            for (size_t k = 0; k < std::size(INTERPOLATORS); k++)
            {
                if (INTERPOLATORS[k].first == interpolator.usage && INTERPOLATORS[k].second == uint32_t(interpolator.usageIndex))
                    interpolantMask |= 1u << k;
            }
        }
        if (!isPixelShader)
            interpolantMask |= 0xFFFFu;  // [rengii] positional: any of the 16 export registers may be written
    }
    if (isPixelShader && interpolantVariant != InterpolantVariant::Velocity)
        interpolantVariant = InterpolantVariant::Trimmed;
    auto declaresInterpolant = [&](size_t k)
    {
        switch (interpolantVariant)
        {
        case InterpolantVariant::Full:
            return true;
        case InterpolantVariant::PositionOnly:
            return false;
        default:
            return ((interpolantMask >> k) & 1u) != 0;
        }
    };
    const bool velocity = interpolantVariant == InterpolantVariant::Velocity;

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void shaderMain(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        for (size_t k = 0; k < std::size(INTERPOLATORS); k++)
        {
            if (!declaresInterpolant(k))
                continue;
            const auto& [usage, usageIndex] = INTERPOLATORS[k];
            println("\t[[vk::location({0})]] in float4 i{1}{2} : {3}{2},", k, USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
        }
        if (velocity)
        {
            println("\t[[vk::location({})]] in float4 iVelCur : VELOCITY0,", std::size(INTERPOLATORS));
            println("\t[[vk::location({})]] in float4 iVelPrev : VELOCITY1,", std::size(INTERPOLATORS) + 1);
        }

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        // [IzunaRecomp] A shader writing colour 0 only gets the alpha-to-mask
        // coverage as colour 0's second blend source (location 0, index 1):
        // explicit locations for both, as DXC wants all or none.
        const bool coverageOutput = (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0) &&
            !(pixelShader->outputs & (PIXEL_SHADER_OUTPUT_COLOR1 | PIXEL_SHADER_OUTPUT_COLOR2 | PIXEL_SHADER_OUTPUT_COLOR3));
        this->coverageOutput = coverageOutput;
        if (coverageOutput)
            out += ",\n\t[[vk::location(0), vk::index(0)]] out float4 oC0 : SV_Target0"
                   ",\n\t[[vk::location(0), vk::index(1)]] out float4 oC0Coverage : SV_Target1";
        else
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
        else  // [rengii] (float 24 depth: written as the epilogue converts it - only ever lower, so early tests stay)
            out += ",\n\tout float oDepth : SV_DepthLessEqual";
        if (velocity)
            out += ",\n\tout float2 oVel : SV_Target1";
    }
    else
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
        // [rengii] No vertex inputs: every vfetch reads raw guest memory
        // (rengiiVertexFetch), so pipelines need no vertex attributes.
        for (uint32_t i = 0; i < 0; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

            out += '\t';

            print("[[vk::location({})]] ", i);

            println("in {0} i{1}{2} : {3}{2},", usageType, USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex), USAGE_SEMANTICS[uint32_t(vertexElement.usage)]);

            vertexElements.emplace(uint32_t(vertexElement.address), vertexElement);
        }

        out += "\tin uint iVertexId : SV_VertexID,\n";  // [rengii] r0.x (rengiiVertexIndex)

        out += "\tout float4 oPos : SV_Position";

        for (size_t k = 0; k < std::size(INTERPOLATORS); k++)
        {
            if (!declaresInterpolant(k))
                continue;
            const auto& [usage, usageIndex] = INTERPOLATORS[k];
            print(",\n\t[[vk::location({0})]] out float4 o{1}{2} : {3}{2}", k, USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
        }
        if (velocity)
        {
            print(",\n\t[[vk::location({})]] out float4 oVelCur : VELOCITY0", std::size(INTERPOLATORS));
            print(",\n\t[[vk::location({})]] out float4 oVelPrev : VELOCITY1", std::size(INTERPOLATORS) + 1);
        }
    }

    out += ")\n";
    out += "{\n";

    if (velocity)
    {
        if (isPixelShader)
        {
            out += "\toVel = (iVelPrev.w > 1e-6) ? (iVelCur.xy / iVelCur.w - iVelPrev.xy / iVelPrev.w) * float2(0.5, -0.5) : float2(8.0, 8.0);\n";
        }
        else
        {
            out += "\toVelCur = 0.0;\n";
            out += "\toVelPrev = 0.0;\n";
            out += "\t[unroll] for (int velIter = 0; velIter < 2; velIter++)\n";
            out += "\t{\n";
        }
    }

    bool definedLoopRegisters[32]{};

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());
                literalConstants.insert(definition->registerIndex + i - (isPixelShader ? 256 : 0));

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                uint32_t loopRegisterIndex = (definition->registerIndex - 8992) / 4 + i;
                println("\tint4 i{} = int4({}, {}, {}, {});",
                    loopRegisterIndex, x, y, z, w);
                if (loopRegisterIndex < 32)
                    definedLoopRegisters[loopRegisterIndex] = true;
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            interpolator.usage = DeclUsage::TexCoord;  // [rengii] positional (see the mask)
            interpolator.usageIndex = uint32_t(interpolator.reg);
            // [IzunaRecomp] (exact where the Xenos interpolator is: rengiiInterpolated)
            println("\tfloat4 r{} = rengiiInterpolated(i{}{});", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            interpolator.usage = DeclUsage::TexCoord;  // [rengii] positional (see the mask)
            interpolator.usageIndex = i;
            interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }
    if (!isPixelShader)
        for (uint32_t i = 0; i < 16; i++)  // [rengii] positional: export register i -> TEXCOORD i
            interpolators.emplace(i, fmt::format("oTexCoord{}", i));  // (no-op where the table had it)

    if (!isPixelShader)
    {
        if (velocity)
            out += "\tif (velIter == 1) oVelPrev = oPos;\n";
        out += "\toPos = 0.0;\n";

        for (size_t k = 0; k < std::size(INTERPOLATORS); k++)
        {
            const auto& [usage, usageIndex] = INTERPOLATORS[k];
            if (interpolantVariant == InterpolantVariant::PositionOnly)
            {
                if ((interpolantMask >> k) & 1u)
                    println("\tfloat4 o{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);
            }
            else if (declaresInterpolant(k))
            {
                println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);
            }
        }

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            if (!isPixelShader)
                registerDeclOffsets[i] = out.size() + 1;
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                // [rengii] (guest pixels at any resolution scale)
                out += "float4((iPos.xy * g_RenderScaleInv2 - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
            }
            else if (!isPixelShader && i == 0)
            {
                out += "float4(float(rengiiVertexIndex(iVertexId)), 0.0, 0.0, 0.0);\n";  // [rengii] as the hardware loads it
            }
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    if (!isPixelShader)
        psDeclOffset = out.size() + 1;
    out += "\tfloat ps = 0.0;\n";
    out += "\tfloat4 rengiiUnsupportedExport = 0.0;\n";  // [rengii] see unsupportedExports
    out += "\tfloat4 rengiieA = 0.0, rengiieM0 = 0.0, rengiieM1 = 0.0, rengiieM2 = 0.0, rengiieM3 = 0.0, rengiieM4 = 0.0;\n";  // [rengii] memexport
    out += "\tfloat4 vpre = 0.0;\n";
    if (isPixelShader)
    {
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    physicalCodeOffset = shader->physicalOffset;

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader->size;
    bool simpleControlFlow = true;

    bool usedLoopRegisters[32]{};

    enum class CfKind : uint8_t { None, Cond, Uncond, LoopStart, LoopEnd };
    struct CfSummary
    {
        CfKind kind = CfKind::None;
        uint32_t target = 0;
    };
    std::vector<CfSummary> cfSummaries;
    std::vector<uint32_t> loopStarts;
    std::unordered_map<uint32_t, uint32_t> loopEndFor;
    uint32_t cfIndex = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                break;

            case ControlFlowOpcode::LoopStart:
                usedLoopRegisters[cfInstr.loopStart.loopId] = true;
                loopStarts.push_back(cfIndex);
                cfSummaries.push_back({ CfKind::LoopStart, 0 });
                break;

            case ControlFlowOpcode::LoopEnd:
                usedLoopRegisters[cfInstr.loopEnd.loopId] = true;
                if (loopStarts.empty())
                    simpleControlFlow = false;
                else
                {
                    loopEndFor[loopStarts.back()] = cfIndex;
                    loopStarts.pop_back();
                }
                cfSummaries.push_back({ CfKind::LoopEnd, 0 });
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.direction || cfInstr.condJmp.address <= cfIndex)
                    simpleControlFlow = false;

                cfSummaries.push_back({
                    cfInstr.condJmp.isUnconditional ? CfKind::Uncond : CfKind::Cond,
                    uint32_t(cfInstr.condJmp.address) });
                break;
            }
            }

            if (cfSummaries.size() == cfIndex)
                cfSummaries.push_back({});

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);

            ++cfIndex;
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    for (uint32_t loopRegister = 0; loopRegister < 32; loopRegister++)
    {
        if (usedLoopRegisters[loopRegister] && !definedLoopRegisters[loopRegister])
        {
            auto namedLoopConstant = int4Constants.find(loopRegister);
            if (namedLoopConstant != int4Constants.end())
                println("\tint4 i{} = {};", loopRegister, namedLoopConstant->second);
            else
                println("\tint4 i{} = int4(g_LoopConstants({}));", loopRegister, loopRegister);
        }
    }

    if (!loopStarts.empty())
        simpleControlFlow = false;

    if (simpleControlFlow)
    {
        const uint32_t instrCount = (instrSize / 12) * 2;
        if (cfSummaries.size() > instrCount)
            cfSummaries.resize(instrCount);

        std::function<bool(uint32_t, uint32_t, uint32_t)> structure =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool
        {
            uint32_t i = lo;
            while (i < hi)
            {
                const CfSummary& summary = cfSummaries[i];
                switch (summary.kind)
                {
                case CfKind::Uncond:
                {
                    if (i + 1 == hi && (summary.target == cont || summary.target == hi))
                        i = hi;
                    else
                        return false;
                    break;
                }
                case CfKind::Cond:
                {
                    const uint32_t t = summary.target;
                    if (t > hi)
                    {
                        if (t != cont)
                            return false;
                        ++ifEndLabels[hi];
                        if (!structure(i + 1, hi, cont))
                            return false;
                        i = hi;
                    }
                    else if (t < hi && t - 1 > i && cfSummaries[t - 1].kind == CfKind::Uncond &&
                             cfSummaries[t - 1].target != t)
                    {
                        const uint32_t merge = cfSummaries[t - 1].target;
                        if (merge == cont || merge == hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[hi];
                            if (!structure(i + 1, t - 1, merge == cont ? cont : hi))
                                return false;
                            if (!structure(t, hi, cont))
                                return false;
                            i = hi;
                        }
                        else if (merge < hi)
                        {
                            elseLabels.insert(t);
                            ++ifEndLabels[merge];
                            if (!structure(i + 1, t - 1, merge))
                                return false;
                            if (!structure(t, merge, merge))
                                return false;
                            i = merge;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        ++ifEndLabels[t];
                        if (!structure(i + 1, t, t == hi ? cont : t))
                            return false;
                        i = t;
                    }
                    break;
                }
                case CfKind::LoopStart:
                {
                    auto loopEnd = loopEndFor.find(i);
                    if (loopEnd == loopEndFor.end() || loopEnd->second >= hi)
                        return false;
                    if (!structure(i + 1, loopEnd->second, loopEnd->second))
                        return false;
                    i = loopEnd->second + 1;
                    break;
                }
                case CfKind::LoopEnd:
                    return false;

                default:
                    ++i;
                    break;
                }
            }
            return true;
        };

        simpleControlFlow = structure(0, instrCount, instrCount);
    }

    if (!simpleControlFlow)
    {
        ifEndLabels.clear();
        elseLabels.clear();
    }

    straightLineFlow = simpleControlFlow;

    if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (!simpleControlFlow)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
                if (elseLabels.count(pc) != 0)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                    indent();
                    out += "else\n";
                    indent();
                    out += "{\n";
                    ++indentation;
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd ||
                                cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);
                {
                    indent();
                    auto findResult = boolConstants.find(cfInstr.condExec.boolAddress);
                    if (findResult != boolConstants.end())
                        println("if ({}{})", cfInstr.condExec.condition ? "" : "!", findResult->second);
                    else
                        println("if ({}BOOL_BIT({}))", cfInstr.condExec.condition ? "" : "!", uint32_t(cfInstr.condExec.boolAddress));
                    indent();
                    out += "{\n";
                    ++indentation;
                    shouldCloseCurlyBracket = true;
                }
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                break;

            case ControlFlowOpcode::LoopStart:
                if (simpleControlFlow)
                {
                    indent();
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else 
                {
                    out += "\t\t\taL = 0;\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
                if (simpleControlFlow)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional)
                {
                    if (!simpleControlFlow)
                    {
                        println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\tcontinue;\n";
                    }
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
                        const bool jumpIfSet = cfInstr.condJmp.condition ^ simpleControlFlow;
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ({}{})", jumpIfSet ? "" : "!", findResult->second);
                        else
                            println("if ({}BOOL_BIT({}))", jumpIfSet ? "" : "!", uint32_t(cfInstr.condJmp.boolAddress));
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            auto instructionCode = code + address * 3;
            
            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            if (shouldReturn)
            {
                if (isPixelShader)
                {
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    out += "\trengiiAlphaTest(oC0.w);\n";  // [rengii] the test's own function (shader_common.h)

                    indent();
                    out += "}";
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_MASK;
                    if (coverageOutput)
                    {
                        // [IzunaRecomp] (the coverage output: 1 unless 2x MSAA's)
                        specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_MASK_MSAA;
                        indent();
                        out += "oC0Coverage = float4(1.0, 1.0, 1.0, 1.0);\n";
                        indent();
                        out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_MASK) { "
                               "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_MASK_MSAA) "
                               "oC0Coverage = rengiiAlphaToMaskCoverage(oC0.w, iPos.xy).xxxx; "
                               "else rengiiAlphaToMask(oC0.w, iPos.xy); }\n";
                    }
                    else
                    {
                        indent();
                        out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_MASK) rengiiAlphaToMask(oC0.w, iPos.xy);\n";
                    }

                    indent();
                    out += "[branch] if (g_PackedDec3 & 0x80000000u) oC0.rgb = linearToPWLGamma(oC0.rgb);\n";
                    indent();
                    out += "[branch] if (g_PackedDec3 & 0x40000000u) oC0 = saturate(oC0);\n";
                    // 2_10_10_10_FLOAT render targets (SharedConstants + 396
                    // bits 24-27, one per target): EDRAM keeps unsigned 7e3
                    // colour (0 to 31.875) and a 2-bit alpha, so the shader's
                    // output is clamped to what they can hold (the renderer
                    // stores them as 16-bit float, which would keep negatives).
                    {
                        const uint32_t outputs = reinterpret_cast<const PixelShader*>(shader)->outputs;
                        const uint32_t bits[4] = {PIXEL_SHADER_OUTPUT_COLOR0, PIXEL_SHADER_OUTPUT_COLOR1,
                                                  PIXEL_SHADER_OUTPUT_COLOR2, PIXEL_SHADER_OUTPUT_COLOR3};
                        for (uint32_t i = 0; i < 4; ++i)
                        {
                            if (!(outputs & bits[i]))
                                continue;
                            indent();
                            out += fmt::format("[branch] if (g_PackedDec3 & {}u) oC{} = clamp(oC{}, 0.0, "
                                               "float4(31.875, 31.875, 31.875, 1.0));\n", 0x01000000u << i, i, i);
                        }
                        // Depth at the 20e4 depth buffer's precision (SPEC_CONSTANT_FLOAT24_DEPTH).
                        indent();
                        if (outputs & PIXEL_SHADER_OUTPUT_DEPTH)
                            out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_FLOAT24_DEPTH) oDepth = rengiiDepthTo24(oDepth);\n";
                        else
                            out += "oDepth = (g_SpecConstants() & SPEC_CONSTANT_FLOAT24_DEPTH) ? rengiiDepthTo24(iPos.z) : iPos.z;\n";
                    }
                }
                else
                {
                    {
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                        out += "\toPos.xyz = oPos.xyz * g_PosScale.xyz + g_PosOffset.xyz * oPos.w;\n";
                        out += "\tif (rengiiRestartVertex) oPos = asfloat(0x7FC00000u).xxxx;\n";  // [rengii] strip restart: its triangles drop
                    }
                }

                if (simpleControlFlow)
                {
                    indent();
                    {
                        if (velocity && !isPixelShader)
                            out += "continue;\n";
                        else
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!simpleControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }
    else
    {
        auto findResult = ifEndLabels.find(pc);
        if (findResult != ifEndLabels.end())
        {
            for (uint32_t i = 0; i < findResult->second; i++)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }
    }

    if (velocity && !isPixelShader)
    {
        out += "\t}\n";
        out += "\toVelCur = oPos;\n";
    }

    if (!isPixelShader)
    {
        uint8_t needed[64]{};
        bool psNeeded = false;
        uint64_t preciseRegisters = 0;
        bool precisePs = false;
        auto need = [&](uint64_t temps)
            {
                for (uint32_t reg = 0; reg < 64; reg++)
                {
                    if ((temps >> reg) & 1)
                        needed[reg] = 0xF;
                }
            };

        for (auto write = vertexAluWrites.rbegin(); write != vertexAluWrites.rend(); ++write)
        {
            const bool vectorHit = (needed[write->vectorDest] & write->vectorMask) != 0;
            const bool scalarHit = (needed[write->scalarDest] & write->scalarMask) != 0;
            if (!write->conditional)
            {
                if (vectorHit)
                    needed[write->vectorDest] &= ~write->vectorMask;
                if (scalarHit)
                    needed[write->scalarDest] &= ~write->scalarMask;
            }

            bool psRead = psNeeded || scalarHit || write->positionReadsPs;
            if (write->scalarOp)
            {
                psNeeded = false;
                if (psRead)
                {
                    if (write->conditional)
                        psNeeded = true;
                    precisePs = true;
                    need(write->scalarSources);
                    if (write->scalarReadsPs)
                        psNeeded = true;
                }
            }
            else
            {
                psNeeded = psRead;
            }

            if (vectorHit)
            {
                if (write->dotOffset != SIZE_MAX)
                    out[write->dotOffset + 3] = 'P';
                else
                    preciseRegisters |= 1ull << write->vectorDest;
                need(write->vectorSources);
            }

            need(write->positionSources);
        }

        if (precisePs && psDeclOffset != 0)
            out.insert(psDeclOffset, "precise ");
        for (int32_t reg = 31; reg >= 0; reg--)
        {
            if (((preciseRegisters >> reg) & 1) && registerDeclOffsets[reg] != 0)
                out.insert(registerDeclOffsets[reg], "precise ");
        }
    }
    if (isPixelShader) rengiiNarrowInterpolantSnaps(out, simpleControlFlow);

    out += "}";
}
