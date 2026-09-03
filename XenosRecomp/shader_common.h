#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
    // Recover raw int16 TEXCOORDs from R16G16(B16A16)_UINT bindings.
    #define SPEC_CONSTANT_SINT_TEXCOORD     (1 << 2)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
// 256-bit boolean register file (BD bool addresses reach ~158), then per-usage 16-bit-pair swap masks.
// Layout: 4 texture dims * 16 samplers * 4 bytes = 256, sampler indices 16*4 = 64 → shared at byte 320.
#define g_Booleans(i)              vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 320 + (i)*4)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 352)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 356)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 364)
#define g_SwappedNormals           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 368)
#define g_SwappedBinormals         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 372)
#define g_SwappedTangents          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 376)
#define g_SwappedBlendWeights      vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 380)
#define g_SwappedPositions         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 384)
#define g_SintTexcoords            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 388)
// Per-slot Xenos TextureSign kUnsignedBiased mask (fetch returns 2c-1 on rgb).
#define g_BiasedTextures           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 392)
// Packed NORMAL/TANGENT/BINORMAL attributes that are DEC3N (Xenos 2_10_10_10)
// rather than 11:11:10. Bits: normal usageIndex 0-7, tangent 8-15, binormal
// 16-23. EOT packs vertex normals 11:11:10 but tangents DEC3N in the same
// mesh; one decode for all three garbled every skinned tangent basis.
#define g_PackedDec3               vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 396)
// Xenos dynamic loop-constant register file (i0..i31): int4(count, start, step, _) per loop.
// Placed after the shared block (ends at 396) at byte 400 == c25. TODO(reeot runtime): upload
// real loop constants here; zero-init makes undefined loops no-op.
#define g_LoopConstants(i)         vk::RawBufferLoad<uint4>(g_PushConstants.SharedConstants + 400 + (i)*16)
#define g_PosScale                 vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 912)
#define g_PosOffset                vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 928)
#else
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#else

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
#define DEFINE_SHARED_CONSTANTS() \
    uint4 g_BooleansArr[2] : packoffset(c20); \
    uint g_SwappedTexcoords : packoffset(c22.x); \
    float2 g_HalfPixelOffset : packoffset(c22.y); \
    float g_AlphaThreshold : packoffset(c22.w); \
    uint g_SwappedNormals : packoffset(c23.x); \
    uint g_SwappedBinormals : packoffset(c23.y); \
    uint g_SwappedTangents : packoffset(c23.z); \
    uint g_SwappedBlendWeights : packoffset(c23.w); \
    uint g_SwappedPositions : packoffset(c24.x); \
    uint g_SintTexcoords : packoffset(c24.y); \
    uint g_BiasedTextures : packoffset(c24.z); \
    uint g_PackedDec3 : packoffset(c24.w); \
    uint4 g_LoopConstantsArr[32] : packoffset(c25); \
    float4 g_PosScale : packoffset(c57); \
    float4 g_PosOffset : packoffset(c58);

#define g_Booleans(i) (g_BooleansArr[(i) / 4][(i) % 4])
// Xenos dynamic loop-constant register file (i0..i31): int4(count, start, step, _) per loop.
// TODO(reeot runtime): upload real loop constants here; zero-init makes undefined loops no-op.
#define g_LoopConstants(i) g_LoopConstantsArr[i]
#else
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);
#endif

uint g_SpecConstants();

#endif

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
// Test Xenos boolean register N in the unified VS(0..127)/PS(128..255) file.
#define BOOL_BIT(n) ((g_Booleans((n) / 32u) & (1u << ((n) & 31u))) != 0)
#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
Texture1D<float4> g_Texture1DDescriptorHeap[] : register(t0, space4);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

#ifdef REEOT_RECOMP
// Xenos TextureSign hook. `biasedMask` is the per-slot kUnsignedBiased mask
// (g_BiasedTextures, filled by the runtime from the live fetch constants'
// dword0 bits 2-9; the red-channel sign selects for the slot the way the
// gamma path's does). Bit set = the console fetch returns 2c-1 on the colour
// channels, which host texture hardware cannot do; EOT uses it on its normal
// maps and its resolved G-buffer normal texture. The mask is passed as an
// argument because the DXIL path declares it in the per-shader space4 cbuffer,
// which is emitted after this header.
// Xenos piecewise-linear gamma, exactly as the console RB applies it (same
// segment constants as rex::graphics::xenos PWLGammaToLinear/LinearToPWLGamma).
// Encode runs on pixel-shader output into k_8_8_8_8_GAMMA render targets
// (ConvertColor0ToGamma; bit 31 of g_PackedDec3); decode runs on kGamma
// fetches of resolve-backed textures (bits 16-31 of g_BiasedTextures), whose
// float mirrors cannot use the sRGB-view path uploads take.
float pwlGammaToLinear1(float gamma)
{
    gamma = saturate(gamma);
    float scale, offset;
    if (gamma >= 96.0 / 255.0)
    {
        if (gamma >= 192.0 / 255.0) { scale = 8.0 / 1024.0; offset = -1024.0; }
        else                        { scale = 4.0 / 1024.0; offset = -256.0; }
    }
    else
    {
        if (gamma >= 64.0 / 255.0)  { scale = 2.0 / 1024.0; offset = -64.0; }
        else                        { scale = 1.0 / 1024.0; offset = 0.0; }
    }
    float lin = gamma * ((255.0 * 1024.0) * scale) + offset;
    lin += trunc(lin * scale);
    return lin * (1.0 / 1023.0);
}

float linearToPWLGamma1(float lin)
{
    lin = saturate(lin);
    float scale, offset;
    if (lin >= 128.0 / 1023.0)
    {
        if (lin >= 512.0 / 1023.0) { scale = 1023.0 / 8.0; offset = 128.0 / 255.0; }
        else                       { scale = 1023.0 / 4.0; offset = 64.0 / 255.0; }
    }
    else
    {
        if (lin >= 64.0 / 1023.0)  { scale = 1023.0 / 2.0; offset = 32.0 / 255.0; }
        else                       { scale = 1023.0;       offset = 0.0; }
    }
    return trunc(lin * scale) * (1.0 / 255.0) + offset;
}

float3 linearToPWLGamma(float3 lin)
{
    return float3(linearToPWLGamma1(lin.r), linearToPWLGamma1(lin.g),
                  linearToPWLGamma1(lin.b));
}

// alphaMask: bits 16-31 of g_SintTexcoords carry the per-slot kUnsignedBiased sign of the
// W component (fetch dword0 bits 8-9). Xenos signs are per component; EOT's DXT5 "AG"
// normal maps carry X in alpha with the biased sign, and Xenia biases all four.
float4 applyFetchSign(float4 value, uint biasedMask, uint alphaMask, uint slotIndex)
{
    if (biasedMask & (1u << slotIndex))
        value.rgb = value.rgb * 2.0 - 1.0;
    if (alphaMask & (1u << (16u + slotIndex)))
        value.a = value.a * 2.0 - 1.0;
    if (biasedMask & (1u << (16u + slotIndex)))
        value.rgb = float3(pwlGammaToLinear1(value.r), pwlGammaToLinear1(value.g),
                           pwlGammaToLinear1(value.b));
    return value;
}
#endif

float4 tfetch1D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float texCoord)
{
    return g_Texture1DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
// Packed normal/tangent/binormal decode; IA binds as R32_UINT so lane .x
// carries the raw bits (asuint recovers them). The GAME picks the packing per
// attribute in its (runtime-patched) vertex declaration: EOT packs vertex
// NORMALs 11:11:10 but TANGENTs DEC3N (Xenos 2_10_10_10, 10-bit snorm x3) in
// the same mesh. `dec3Mask` bit `slotCode` (normal usageIndex 0-7, tangent
// 8-15, binormal 16-23, filled by the runtime from the live fetch layout)
// selects the DEC3N decode; clear = 11:11:10 as before.
float4 tfetchR11G11B10(uint dec3Mask, float4 value, uint slotCode)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        // w = 1.0: the Xenos vfetch of a 3-component packed format fills the
        // missing lane with one, and the game's skinned VS depends on it --
        // the tangent is skinned with 4-component dots (w picks up the bone
        // translation row) and the binormal is cross(t, n) * t.w. Returning 0
        // here skinned every character's tangent without translation and
        // zeroed the binormal outright: broken TBN -> wrong G-buffer normals
        // -> every deferred light shades characters wrong (the grey-cyan hue
        // plus the missing warm fill on the title screen).
        uint v = asuint(value.x);
        if (dec3Mask & (1u << slotCode))
        {
            int3 s = int3(v << 22, v << 12, v << 2) >> 22;
            return float4(max(float3(s) / 511.0, -1.0), 1.0);
        }
        return float4(
            (v & 0x00000400 ? -1.0 : 0.0) + ((v & 0x3FF) / 1024.0),
            (v & 0x00200000 ? -1.0 : 0.0) + (((v >> 11) & 0x3FF) / 1024.0),
            (v & 0x80000000 ? -1.0 : 0.0) + (((v >> 22) & 0x1FF) / 512.0),
            1.0);
    }
    return value;
}

// Undo the engine bswap32 16-bit-pair swap (.yxwz) for any 16-bit-packed semantic flagged in the mask.
float4 swapFloats(uint swappedMask, float4 value, uint semanticIndex)
{
    return (swappedMask & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}

// Recover X360 integer-cast-to-float TEXCOORDs from R16G16(B16A16)_UINT bindings (sign-extend low 16 bits).
float4 sintTexcoord(uint mask, float4 value, uint semanticIndex)
{
    if ((mask & (1u << semanticIndex)) != 0)
    {
        int4 si = (int4(asuint(value)) << 16) >> 16;
        return float4(si);
    }
    return value;
}
#else
float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}
#endif

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
