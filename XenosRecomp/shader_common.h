#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

// SPIR-V vertex input locations, shared with reblue's host input-layout
// builder. A semantic missing here gets no location from the emitter.
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,  0,  0) \
    X(Position,  1,  1) \
    X(Position,  2,  2) \
    X(Position,  3,  3) \
    X(Position,  4,  4) \
    X(Normal,    0,  5) \
    X(Tangent,   0,  6) \
    X(TexCoord,  0,  7) \
    X(TexCoord,  1,  8) \
    X(TexCoord,  2,  9) \
    X(Color,     0, 10)

// SPEC_CONSTANTS_ONLY keeps the HLSL below out of host C++ TUs, which IntelliSense would otherwise parse.
// MSL defines __cplusplus, so __air__ has to open this block explicitly.
#if (defined(__air__) || !defined(__cplusplus) || defined(__INTELLISENSE__)) \
    && !defined(SHADER_COMMON_SPEC_CONSTANTS_ONLY)

#ifdef __air__

#include <metal_stdlib>

using namespace metal;

// HLSL spellings, so the helpers below and the recompiler's emitted bodies read
// the same in both languages.
static inline uint   asuint(float  v) { return as_type<uint  >(v); }
static inline uint2  asuint(float2 v) { return as_type<uint2 >(v); }
static inline uint3  asuint(float3 v) { return as_type<uint3 >(v); }
static inline uint4  asuint(float4 v) { return as_type<uint4 >(v); }
static inline float  asfloat(uint  v) { return as_type<float >(v); }
static inline float2 asfloat(uint2 v) { return as_type<float2>(v); }
static inline float3 asfloat(uint3 v) { return as_type<float3>(v); }
static inline float4 asfloat(uint4 v) { return as_type<float4>(v); }

template<typename T> static inline T frac(T v) { return fract(v); }
template<typename T> static inline T rcp(T v)  { return 1.0f / v; }
template<typename T> static inline T ddx(T v)  { return dfdx(v); }
template<typename T> static inline T ddy(T v)  { return dfdy(v); }
template<typename T> static inline T lerp(T a, T b, T t) { return mix(a, b, t); }

static inline void clip(float v)  { if (v < 0.0f) discard_fragment(); }
static inline void clip(float4 v) { if (any(v < 0.0f)) discard_fragment(); }

// HLSL takes the condition first, metal::select takes it last. Getting this
// backwards silently picks the wrong branch, so route every call through here.
// decltype(a + b) promotes the scalar when only one side is a vector.
template<typename C, typename A, typename B>
static inline auto hlslSelect(C cond, A a, B b)
{
    typedef decltype(a + b) R;
    return metal::select(R(b), R(a), cond);
}
#define select(c, a, b) hlslSelect(c, a, b)

// metal_stdlib's FLT_MIN is the smallest positive normal, but the emitter uses
// FLT_MIN as a lower clamp bound, which in HLSL is the most negative float.
#undef FLT_MIN
#undef FLT_MAX
#define FLT_MIN asfloat(0xff7fffffu)
#define FLT_MAX asfloat(0x7f7fffffu)

// HLSL's inout becomes a thread reference in MSL.
#define INOUT(T) thread T&

#else

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#define INOUT(T) inout T

#endif

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

// The recompiler emits its constant accessors through these, so the
// buffer-address path is shared between SPIR-V and Metal.
#define BUFFER_LOAD_FLOAT4(address) vk::RawBufferLoad<float4>(address, 0x10)
#define BUFFER_LOAD_UINT(address)   vk::RawBufferLoad<uint>(address)

#ifdef REBLUE_RECOMP
// 256-bit boolean register file (BD bool addresses reach ~158), then per-usage 16-bit-pair swap masks.
#define g_Booleans(i)              vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256 + (i)*4)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 288)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 292)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 300)
#define g_SwappedNormals           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 304)
#define g_SwappedBinormals         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 308)
#define g_SwappedTangents          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 312)
#define g_SwappedBlendWeights      vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 316)
#define g_SwappedPositions         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 320)
#define g_SintTexcoords            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 324)
#define g_ShadowPcfScale           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 328)
#else
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#elif defined(__air__)

// Mirrors the __spirv__ branch above; offsets are reblue's shared-constant
// layout and must stay in step with it. The heap/push-constant names below are
// entry-point parameters, which the recompiler declares with these exact names.
struct PushConstants
{
    ulong VertexShaderConstants;
    ulong PixelShaderConstants;
    ulong SharedConstants;
};

#define BUFFER_LOAD_FLOAT4(address) (*reinterpret_cast<device const float4*>(address))
#define BUFFER_LOAD_UINT(address)   (*reinterpret_cast<device const uint*>(address))

#ifdef REBLUE_RECOMP
// g_HalfPixelOffset lands on a 4-byte offset, so it needs the packed type.
#define g_Booleans(i)              (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 256 + (i) * 4))
#define g_SwappedTexcoords         (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 288))
#define g_HalfPixelOffset          float2(*reinterpret_cast<device const packed_float2*>(g_PushConstants.SharedConstants + 292))
#define g_AlphaThreshold           (*reinterpret_cast<device const float*>(g_PushConstants.SharedConstants + 300))
#define g_SwappedNormals           (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 304))
#define g_SwappedBinormals         (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 308))
#define g_SwappedTangents          (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 312))
#define g_SwappedBlendWeights      (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 316))
#define g_SwappedPositions         (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 320))
#define g_SintTexcoords            (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 324))
#define g_ShadowPcfScale           (*reinterpret_cast<device const float*>(g_PushConstants.SharedConstants + 328))
#else
#define g_Booleans                 (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 256))
#define g_SwappedTexcoords         (*reinterpret_cast<device const uint*>(g_PushConstants.SharedConstants + 260))
#define g_HalfPixelOffset          float2(*reinterpret_cast<device const packed_float2*>(g_PushConstants.SharedConstants + 264))
#define g_AlphaThreshold           (*reinterpret_cast<device const float*>(g_PushConstants.SharedConstants + 272))
#endif

// Index 0 matches plume's MetalShader::createFunction, which supplies this as a
// function constant per pipeline.
constant uint g_SpecConstantsValue [[function_constant(0)]];
static inline uint g_SpecConstants()
{
    return is_function_constant_defined(g_SpecConstantsValue) ? g_SpecConstantsValue : 0u;
}

#else

#ifdef REBLUE_RECOMP
#define DEFINE_SHARED_CONSTANTS() \
    uint4 g_BooleansArr[2] : packoffset(c16); \
    uint g_SwappedTexcoords : packoffset(c18.x); \
    float2 g_HalfPixelOffset : packoffset(c18.y); \
    float g_AlphaThreshold : packoffset(c18.w); \
    uint g_SwappedNormals : packoffset(c19.x); \
    uint g_SwappedBinormals : packoffset(c19.y); \
    uint g_SwappedTangents : packoffset(c19.z); \
    uint g_SwappedBlendWeights : packoffset(c19.w); \
    uint g_SwappedPositions : packoffset(c20.x); \
    uint g_SintTexcoords : packoffset(c20.y); \
    float g_ShadowPcfScale : packoffset(c20.z);

#define g_Booleans(i) (g_BooleansArr[(i) / 4][(i) % 4])
#else
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);
#endif

uint g_SpecConstants();

#endif

#ifdef REBLUE_RECOMP
// Test Xenos boolean register N in the unified VS(0..127)/PS(128..255) file.
#define BOOL_BIT(n) ((g_Booleans((n) / 32u) & (1u << ((n) & 31u))) != 0)
#endif

#ifdef __air__

// Argument-buffer elements. MSL has no global resources, so everything that
// touches a heap takes it as a parameter and gets a call-site macro at the
// bottom of this header to put the bare HLSL spelling back.
struct Texture2DDescriptorHeap { texture2d<float> tex; };
struct Texture3DDescriptorHeap { texture3d<float> tex; };
struct TextureCubeDescriptorHeap { texturecube<float> tex; };
struct SamplerDescriptorHeap { sampler samp; };

uint2 getTexture2DDimensions(texture2d<float> texture)
{
    return uint2(texture.get_width(), texture.get_height());
}

float4 tfetch2DHeaps(constant Texture2DDescriptorHeap* textureHeap, constant SamplerDescriptorHeap* samplerHeap,
    uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    texture2d<float> texture = textureHeap[resourceDescriptorIndex].tex;
    return texture.sample(samplerHeap[samplerDescriptorIndex].samp, texCoord + offset / float2(getTexture2DDimensions(texture)));
}

float2 getWeights2DHeaps(constant Texture2DDescriptorHeap* textureHeap, constant SamplerDescriptorHeap* samplerHeap,
    uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    texture2d<float> texture = textureHeap[resourceDescriptorIndex].tex;
    return select(isnan(texCoord), 0.0, frac(texCoord * float2(getTexture2DDimensions(texture)) + offset - 0.5));
}


#ifdef REBLUE_RECOMP
// Bilinear-filtered shadow compare over the four neighboring depth texels.
// Metal counterpart of the HLSL shadowCmp2D below; Load() becomes read().
float shadowCmp2DHeaps(constant Texture2DDescriptorHeap* textureHeap,
    uint resourceDescriptorIndex, float2 texCoord, float ref)
{
    texture2d<float> texture = textureHeap[resourceDescriptorIndex].tex;
    int2 dimensions = int2(getTexture2DDimensions(texture));
    float2 coord = texCoord * float2(dimensions) - 0.5f;
    float2 weights = frac(coord);
    int2 base = int2(floor(coord));
    int2 c0 = clamp(base, int2(0, 0), dimensions - 1);
    int2 c1 = clamp(base + 1, int2(0, 0), dimensions - 1);
    float4 depths = float4(
        texture.read(uint2(c0), 0).x,
        texture.read(uint2(c1.x, c0.y), 0).x,
        texture.read(uint2(c0.x, c1.y), 0).x,
        texture.read(uint2(c1), 0).x);
    // MSL will not widen the bool4 compare implicitly.
    float4 taps = float4(depths > ref);
    return lerp(lerp(taps.x, taps.y, weights.x), lerp(taps.z, taps.w, weights.x), weights.y);
}
#endif

#else

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
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

#ifdef REBLUE_RECOMP
// Bilinear-filtered shadow compare over the four neighboring depth texels.
float shadowCmp2D(uint resourceDescriptorIndex, float2 texCoord, float ref)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    int2 dimensions = int2(getTexture2DDimensions(texture));
    float2 coord = texCoord * dimensions - 0.5;
    float2 weights = frac(coord);
    int2 base = int2(floor(coord));
    int2 c0 = clamp(base, int2(0, 0), dimensions - 1);
    int2 c1 = clamp(base + 1, int2(0, 0), dimensions - 1);
    float4 taps = float4(
        texture.Load(int3(c0, 0)).x,
        texture.Load(int3(c1.x, c0.y, 0)).x,
        texture.Load(int3(c0.x, c1.y, 0)).x,
        texture.Load(int3(c1, 0)).x) > ref;
    return lerp(lerp(taps.x, taps.y, weights.x), lerp(taps.z, taps.w, weights.x), weights.y);
}
#endif
#endif

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

#ifdef __air__

float4 tfetch2DBicubicHeaps(constant Texture2DDescriptorHeap* textureHeap, constant SamplerDescriptorHeap* samplerHeap,
    uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    texture2d<float> texture = textureHeap[resourceDescriptorIndex].tex;
    sampler samplerState = samplerHeap[samplerDescriptorIndex].samp;
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
        g0(fy) * (g0x * texture.sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3DHeaps(constant Texture3DDescriptorHeap* textureHeap, constant SamplerDescriptorHeap* samplerHeap,
    uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return textureHeap[resourceDescriptorIndex].tex.sample(samplerHeap[samplerDescriptorIndex].samp, texCoord);
}

#else

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

#endif

struct CubeMapData
{
	#ifdef REBLUE_RECOMP
    // BD's cube-shadow PCF issues up to 9 cube ops per shader (bd_mirror_cs_ps /
    // bd_glass_cs_ps); overflowing this array drops the stored directions and
    // every tfetchCube reads OOB -> point-light shadows vanish.
    float3 cubeMapDirections[9];
    #else
    float3 cubeMapDirections[2];
    #endif
    uint cubeMapIndex;
};

#ifdef __air__

float4 tfetchCubeHeaps(constant TextureCubeDescriptorHeap* textureHeap, constant SamplerDescriptorHeap* samplerHeap,
    uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, INOUT(CubeMapData) cubeMapData)
{
    // MSL will not index an array with a float the way HLSL does.
    return textureHeap[resourceDescriptorIndex].tex.sample(samplerHeap[samplerDescriptorIndex].samp,
        cubeMapData.cubeMapDirections[uint(texCoord.z)]);
}

#else

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, INOUT(CubeMapData) cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

#endif

#ifdef REBLUE_RECOMP
// DEC3N normal decode; IA binds as R32_UINT so lane .x carries the raw bits (asuint recovers them).
float4 tfetchR11G11B10(float4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        uint v = asuint(value.x);
        return float4(
            (v & 0x00000400 ? -1.0 : 0.0) + ((v & 0x3FF) / 1024.0),
            (v & 0x00200000 ? -1.0 : 0.0) + (((v >> 11) & 0x3FF) / 1024.0),
            (v & 0x80000000 ? -1.0 : 0.0) + (((v >> 22) & 0x1FF) / 512.0),
            0.0);
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

float4 cube(float4 value, INOUT(CubeMapData) cubeMapData)
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

#ifdef __air__

float2 getPixelCoordHeaps(constant Texture2DDescriptorHeap* textureHeap, uint resourceDescriptorIndex, float2 texCoord)
{
    return float2(getTexture2DDimensions(textureHeap[resourceDescriptorIndex].tex)) * texCoord;
}

#else

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

#endif

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#ifdef __air__
// Restores the bare HLSL spellings for the emitted body, binding the heaps from
// the entry point's parameters. Defined last so the helpers above are unaffected.
#define tfetch2D(r, s, texCoord, offset)        tfetch2DHeaps(g_Texture2DDescriptorHeap, g_SamplerDescriptorHeap, r, s, texCoord, offset)
#define getWeights2D(r, s, texCoord, offset)    getWeights2DHeaps(g_Texture2DDescriptorHeap, g_SamplerDescriptorHeap, r, s, texCoord, offset)
#define tfetch2DBicubic(r, s, texCoord, offset) tfetch2DBicubicHeaps(g_Texture2DDescriptorHeap, g_SamplerDescriptorHeap, r, s, texCoord, offset)
#define tfetch3D(r, s, texCoord)                tfetch3DHeaps(g_Texture3DDescriptorHeap, g_SamplerDescriptorHeap, r, s, texCoord)
#define tfetchCube(r, s, texCoord, cubeMapData) tfetchCubeHeaps(g_TextureCubeDescriptorHeap, g_SamplerDescriptorHeap, r, s, texCoord, cubeMapData)
#define getPixelCoord(r, texCoord)              getPixelCoordHeaps(g_Texture2DDescriptorHeap, r, texCoord)
#ifdef REBLUE_RECOMP
#define shadowCmp2D(r, texCoord, ref)           shadowCmp2DHeaps(g_Texture2DDescriptorHeap, r, texCoord, ref)
#endif
#endif

#endif

#endif
