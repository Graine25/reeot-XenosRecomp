#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)
#define SPEC_CONSTANT_SPECULAR_AA       (1 << 5)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#ifdef REEOT_RECOMP
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

#ifdef REEOT_RECOMP
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
#define g_BiasedTextures           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 392)
#define g_PackedDec3               vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 396)
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
#define g_LoopConstants(i) g_LoopConstantsArr[i]
#else
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);
#endif

#ifdef REEOT_RECOMP
cbuffer SpecConstants : register(b4, space4)
{
    uint g_SpecConstantsValue;
};
#define g_SpecConstants() g_SpecConstantsValue
#else
uint g_SpecConstants();
#endif

#endif

#ifdef REEOT_RECOMP
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

// Specular AA for the light pre-pass (eot_specular_aa): the resolved normal renormalized, the
// lobe widened by its lost length (Toksvig) and its screen-space variance (Tokuyoshi and
// Kaplanyan 2021), and the epilogue keeps the narrow lobe's energy.
static float g_specularAAEnergy = 1.0;
float4 specularAAFetch(float4 value)
{
    [branch] if (!(g_SpecConstants() & SPEC_CONSTANT_SPECULAR_AA))
        return value;
    const float len = length(value.xyz);
    const float3 n = value.xyz * (len > 1e-4 ? rcp(len) : 0.0);
    const float r = saturate(len);
    const float power = max(value.w * 127.0, 0.0);
    const float toksvig = r * power / max(r + power * (1.0 - r), 1e-4);
    const float3 dndx = ddx(n);
    const float3 dndy = ddy(n);
    const float kernel = min(0.15915494 * (dot(dndx, dndx) + dot(dndy, dndy)), 0.18);
    const float alpha2 = min(2.0 / (toksvig + 2.0) + kernel, 1.0);
    const float filtered = max(2.0 / alpha2 - 2.0, 0.0);
    g_specularAAEnergy = (filtered + 1.0) / (power + 1.0);
    return float4(n, filtered / 127.0);
}

// The fast twin's hook: the sign selects without the gamma decode.
float4 applyFetchSignFast(float4 value, uint biasedMask, uint alphaMask, uint slotIndex)
{
    if (biasedMask & (1u << slotIndex))
        value.rgb = value.rgb * 2.0 - 1.0;
    if (alphaMask & (1u << (16u + slotIndex)))
        value.a = value.a * 2.0 - 1.0;
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

#ifdef REEOT_RECOMP
float4 tfetchR11G11B10(uint dec3Mask, float4 value, uint slotCode)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        uint v = asuint(value.x);
        if (dec3Mask & (1u << slotCode))
        {
            int4 s = int4(v << 22, v << 12, v << 2, v) >> int4(22, 22, 22, 30);
            return max(float4(s) / float4(511.0, 511.0, 511.0, 1.0), -1.0);
        }
        return float4(
            (v & 0x00000400 ? -1.0 : 0.0) + ((v & 0x3FF) / 1024.0),
            (v & 0x00200000 ? -1.0 : 0.0) + (((v >> 11) & 0x3FF) / 1024.0),
            (v & 0x80000000 ? -1.0 : 0.0) + (((v >> 22) & 0x1FF) / 512.0),
            1.0);
    }
    return value;
}

float4 swapFloats(uint swappedMask, float4 value, uint semanticIndex)
{
    return (swappedMask & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}

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

void eotTwoSum(float a, float b, out float s, out float e)
{
    precise float sum = a + b;
    precise float bv = sum - a;
    precise float err = (a - (sum - bv)) + (b - bv);
    s = sum;
    e = err;
}

void eotTwoProduct(float a, float b, out float p, out float e)
{
    precise float ca = 4097.0 * a;
    precise float ah = ca - (ca - a);
    precise float al = a - ah;
    precise float cb = 4097.0 * b;
    precise float bh = cb - (cb - b);
    precise float bl = b - bh;
    precise float prod = a * b;
    precise float err = ((ah * bh - prod) + ah * bl + al * bh) + al * bl;
    p = prod;
    e = err;
}

float dotF(float4 a, float4 b) { return dot(a, b); }
float dotF(float3 a, float3 b) { return dot(a, b); }

float dotP(float4 a, float4 b)
{
    float p, s, h, r, q;
    eotTwoProduct(a.x, b.x, p, s);
    eotTwoProduct(a.y, b.y, h, r);
    eotTwoSum(p, h, p, q);
    precise float s1 = s + (q + r);
    eotTwoProduct(a.z, b.z, h, r);
    eotTwoSum(p, h, p, q);
    precise float s2 = s1 + (q + r);
    eotTwoProduct(a.w, b.w, h, r);
    eotTwoSum(p, h, p, q);
    precise float s3 = s2 + (q + r);
    precise float result = p + s3;
    return isfinite(result) ? result : dot(a, b);
}

float dotP(float3 a, float3 b)
{
    float p, s, h, r, q;
    eotTwoProduct(a.x, b.x, p, s);
    eotTwoProduct(a.y, b.y, h, r);
    eotTwoSum(p, h, p, q);
    precise float s1 = s + (q + r);
    eotTwoProduct(a.z, b.z, h, r);
    eotTwoSum(p, h, p, q);
    precise float s2 = s1 + (q + r);
    precise float result = p + s2;
    return isfinite(result) ? result : dot(a, b);
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
