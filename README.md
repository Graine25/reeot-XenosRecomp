# XenosRecomp for ReNGII

The `rengii` branch translates Ninja Gaiden II shader containers to HLSL and a
Vulkan 1.2 shader cache. Its NGII paths come from
[IzunaRecomp's vendored fork](https://github.com/YaibaKamikaze/IzunaRecomp/tree/3045e0629a8d693ee1acdab3647e68d6e2b644fe/third_party/xenosrecomp),
identified there as `fb51ac62fc640d07f5b27f154e81c7f7b785a4a2` (MIT).
The compiler and generated helpers use `RENGII`/`rengii` names.

This branch carries over raw and mini vertex fetches, vertex textures, all 32
texture slots, global boolean and loop constants, shader literals, sampler/LOD
overrides, memory exports, float24 depth, alpha testing and alpha-to-mask.
The default cache uses trimmed interpolants. The optional full, position-only
and velocity translation paths remain available in `ShaderRecompiler`.

## Build

A C++20 compiler, CMake 3.20 and the nested dependencies are required.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target XenosRecomp --parallel 6
```

## Generate shaders

Translate one complete container to HLSL:

```sh
build/XenosRecomp/XenosRecomp shader.container.bin shader.hlsl XenosRecomp/shader_common.h
```

Scan a directory, deduplicate complete containers by XXH3, compile them with
DXC, and write a cache plus optional HLSL files:

```sh
build/XenosRecomp/XenosRecomp containers shader_notes/shader_cache.cpp XenosRecomp/shader_common.h shader_notes/hlsl
```

This writes `shader_cache.cpp`, its standalone `shader_cache.h`, and
`shader_cache.bin`. Cache entries are sorted by hash and retain the shader
stage, specialization mask, vertex-fetch instruction addresses, physical code
offset, constant bound, texture/filter masks, memory-export use and coverage
output. The HLSL entry point is `shaderMain`; profiles are `vs_6_6`/`ps_6_6`.
Vertex compilation applies Vulkan Y inversion.

The `.bin` is the same zstd-compressed bytes embedded in the C++ array.
Decompress with zstd, then use each entry's `spirvOffset`/`spirvSize` to decode
its SMOL-V module. Offsets address the decompressed **SMOL-V** stream. Decode
a module before passing its SPIR-V words to Vulkan.

## Renderer contract

`shader_common.h` defines the host ABI. Five 64-bit push-constant addresses
supply vertex constants, pixel constants, shared constants, guest memory and
fetch constants. Shared constants include the global boolean file at byte 320,
loop constants at 400, texture/sampler indices at 1024, and draw-time vertex
fetch records at 1664. These are host offsets, not guest device offsets.

D3D patches vertex fetch instructions when a declaration is bound. Capture the
patched fields per draw; the original shader container is insufficient.
`ShaderVertexFetch::instruction` is in 12-byte units from the physical code
start, after `physicalCodeOffset`. Mini fetches inherit the previous full
fetch's index, slot and stride. Texture filter masks are aggregate requirements;
conflicting requests on one slot still need per-instruction handling.

`coverageOutput` means location 0, blend index 1. It needs dual-source blending,
not an extra render target. Memory-export writes need the correct guest-memory
mapping and synchronization. Small export formats remain incomplete. The NGII
raw-memory paths currently require SPIR-V; DXIL cache generation is disabled.
Valid SPIR-V alone does not establish correct rendering or device support.

Thanks to Xenia and the original XenosRecomp authors for the instruction and
container research.
