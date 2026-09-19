#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"

#include <mutex>
#include <vector>

static std::unique_ptr<uint8_t[]> readAllBytes(const char* filePath, size_t& fileSize)
{
    FILE* file = fopen(filePath, "rb");
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    auto data = std::make_unique<uint8_t[]>(fileSize);
    fread(data.get(), 1, fileSize, file);
    fclose(file);
    return data;
}

static void writeAllBytes(const char* filePath, const void* data, size_t dataSize)
{
    FILE* file = fopen(filePath, "wb");
    fwrite(data, 1, dataSize, file);
    fclose(file);
}

struct RecompiledShader
{
    uint8_t* data = nullptr;
    IDxcBlob* dxil = nullptr;
    std::vector<uint8_t> spirv;
    uint32_t specConstantsMask = 0;
// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
    std::vector<VertexFetchLayoutRecord> vertexLayout;  // empty for pixel shaders
    uint32_t vfetchCodeOffset = 0;                      // instruction base in the physical part
    uint32_t usesFloatConstants = 0;                    // 0 = window-space VS (no WVP constants)
    uint32_t interpolantMask = 0;                       // ShaderRecompiler::interpolantMask
    // VS only: the full and the position-only interpolant variants
    // (shader_recompiler.h); empty when the variant failed to compile, the
    // runtime then uses the trimmed blob.
    IDxcBlob* fullDxil = nullptr;
    std::vector<uint8_t> fullSpirv;
    IDxcBlob* posDxil = nullptr;
    std::vector<uint8_t> posSpirv;
    // VS and PS: the motion-vector variant (shader_recompiler.h Velocity);
    // empty when it failed to compile, the runtime then draws the material
    // without a motion vector.
    IDxcBlob* velDxil = nullptr;
    std::vector<uint8_t> velSpirv;
#endif
};

// Per-shader recompile failures, collected from the parallel loop and reported before exit.
struct ShaderFailure
{
    XXH64_hash_t hash;
    std::string reason;
};

static std::mutex g_failureMutex;
static std::vector<ShaderFailure> g_failures;
// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
// A vertex shader whose full or position-only variant failed keeps its
// trimmed blob for that variant; reported, not fatal.
static std::vector<ShaderFailure> g_variantFailures;
#endif

int main(int argc, char** argv)
{
#ifndef XENOS_RECOMP_INPUT
    if (argc < 4)
    {
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path]");
        return 0;
    }
#endif

    const char* input =
#ifdef XENOS_RECOMP_INPUT 
        XENOS_RECOMP_INPUT
#else
        argv[1]
#endif
    ;

    const char* output =
#ifdef XENOS_RECOMP_OUTPUT 
        XENOS_RECOMP_OUTPUT
#else
        argv[2]
#endif
        ;
    
    const char* includeInput =
#ifdef XENOS_RECOMP_INCLUDE_INPUT
        XENOS_RECOMP_INCLUDE_INPUT
#else
        argv[3]
#endif
        ;

    size_t includeSize = 0;
    auto includeData = readAllBytes(includeInput, includeSize);
    std::string_view include(reinterpret_cast<const char*>(includeData.get()), includeSize);

    // Optional per-shader HLSL dump mode: scans the input (file or directory) for shader
    // containers and writes each recompiled shader as <output>/<hash>.hlsl. The hash matches
    // the XXH3-64 lookup key used by the runtime shader cache, so the native renderer can map
    // a guest shader straight to its HLSL by hash.
    bool hlslDump = false;
    bool dxilDump = false;
    bool velocityDump = false; // --hlsl-vel: dump the Velocity variant instead
#ifndef XENOS_RECOMP_INPUT
    if (argc >= 5 && std::string_view(argv[4]) == "--hlsl")
        hlslDump = true;
    if (argc >= 5 && std::string_view(argv[4]) == "--hlsl-vel")
        hlslDump = velocityDump = true;
    if (argc >= 5 && std::string_view(argv[4]) == "--dxil")
        dxilDump = true;
#endif

    if (std::filesystem::is_directory(input) || hlslDump || dxilDump)
    {
        std::vector<std::unique_ptr<uint8_t[]>> files;
        std::map<XXH64_hash_t, RecompiledShader> shaders;

        // Pre-dedup tallies of valid shader containers, split by type, for cross-referencing
        // against the raw magic-byte occurrence counts (10 2A 11 00 = pixel, 10 2A 11 01 = vertex).
        size_t foundPixel = 0;
        size_t foundVertex = 0;

        // Gather the files to scan: every file under a directory, or just the single input file.
        std::vector<std::string> inputPaths;
        if (std::filesystem::is_directory(input))
        {
            for (auto& file : std::filesystem::recursive_directory_iterator(input))
                if (!std::filesystem::is_directory(file))
                    inputPaths.push_back(file.path().string());
        }
        else
        {
            inputPaths.push_back(input);
        }

        for (auto& path : inputPaths)
        {
            size_t fileSize = 0;
            auto fileData = readAllBytes(path.c_str(), fileSize);
            bool foundAny = false;

            for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
            {
                auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
                size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

                if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                    dataSize <= (fileSize - i) &&
                    shaderContainer->field1C == 0 &&
                    shaderContainer->field20 == 0)
                {
                    if ((shaderContainer->flags & 0xFF) == 0)
                        ++foundPixel;
                    else
                        ++foundVertex;

                    XXH64_hash_t hash = XXH3_64bits(shaderContainer, dataSize);
                    auto shader = shaders.try_emplace(hash);
                    if (shader.second)
                    {
                        shader.first->second.data = fileData.get() + i;
                        foundAny = true;
                    }

                    i += dataSize;
                }
                else
                {
                    i += sizeof(uint32_t);
                }
            }

            if (foundAny)
                files.emplace_back(std::move(fileData));
        }

        if (hlslDump)
        {
            std::filesystem::create_directories(output);

            std::atomic<uint32_t> dumped = 0;
            std::atomic<uint32_t> failed = 0;
            std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
                {
                    const XXH64_hash_t hash = hashShaderPair.first;

                    thread_local ShaderRecompiler recompiler;
                    recompiler = {};
#ifdef REEOT_RECOMP
                    if (velocityDump)
                        recompiler.interpolantVariant = ShaderRecompiler::InterpolantVariant::Velocity;
#endif
                    try
                    {
                        recompiler.recompile(hashShaderPair.second.data, include);
                    }
                    catch (...)
                    {
                        ++failed;
                        return;
                    }

                    std::string fileName = fmt::format("{}/{:016x}.hlsl", output, hash);
                    writeAllBytes(fileName.c_str(), recompiler.out.data(), recompiler.out.size());
                    ++dumped;
                });

            // Machine-parseable stats line for cross-referencing (found = valid containers
            // pre-dedup; unique = distinct hashes; dumped/failed = recompile results).
            fmt::println("STATS found_pixel={} found_vertex={} found_total={} unique={} dumped={} failed={}",
                foundPixel, foundVertex, foundPixel + foundVertex, shaders.size(), dumped.load(), failed.load());
            fmt::println("Dumped {} HLSL shaders to {} ({} failed to recompile).", dumped.load(), output, failed.load());
            return 0;
        }

        if (dxilDump)
        {
            std::filesystem::create_directories(output);

            std::atomic<uint32_t> dumped = 0;
            std::atomic<uint32_t> failed = 0;
            std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
                {
                    const XXH64_hash_t hash = hashShaderPair.first;

                    thread_local ShaderRecompiler recompiler;
                    recompiler = {};
                    try
                    {
                        recompiler.recompile(hashShaderPair.second.data, include);
                    }
                    catch (...)
                    {
                        ++failed;
                        return;
                    }

                    // Spec constants are normally a runtime DXIL-link (g_SpecConstants()
                    // is left unimplemented). For the plain-DXIL path, bake them to 0
                    // (default variant: alpha-test off, R11G11B10 off, ...) so the
                    // spec-constant shaders compile standalone. TODO: real per-mask link.
                    std::string src = recompiler.out;
                    {
                        const std::string decl = "uint g_SpecConstants();";
                        size_t pos = src.find(decl);
                        if (pos != std::string::npos)
                            src.replace(pos, decl.size(), "uint g_SpecConstants() { return 0; }");
                    }

                    thread_local DxcCompiler dxcCompiler;
                    IDxcBlob* dxil = dxcCompiler.compile(src, recompiler.isPixelShader, false, false);
                    if (dxil == nullptr)
                    {
                        ++failed;
                        return;
                    }

                    std::string fileName = fmt::format("{}/{:016x}.dxil", output, hash);
                    writeAllBytes(fileName.c_str(), dxil->GetBufferPointer(), dxil->GetBufferSize());
                    dxil->Release();
                    ++dumped;
                });

            fmt::println("STATS found_total={} unique={} dumped={} failed={}",
                foundPixel + foundVertex, shaders.size(), dumped.load(), failed.load());
            fmt::println("Dumped {} DXIL shaders to {} ({} failed).", dumped.load(), output, failed.load());
            return 0;
        }

        std::atomic<uint32_t> progress = 0;

        std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
            {
                auto& shader = hashShaderPair.second;
                const XXH64_hash_t hash = hashShaderPair.first;

                auto recordFailure = [hash](const char* reason)
                {
                    std::lock_guard<std::mutex> lock(g_failureMutex);
                    g_failures.push_back({hash, reason});
                };

                thread_local ShaderRecompiler recompiler;
                recompiler = {};
                recompiler.recompile(shader.data, include);

                shader.specConstantsMask = recompiler.specConstantsMask;
// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
                shader.vertexLayout = recompiler.vertexLayout;
                shader.vfetchCodeOffset = recompiler.physicalCodeOffset;
                shader.usesFloatConstants = recompiler.usesFloatConstants ? 1 : 0;
#endif

                thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                if (shader.dxil == nullptr)
                {
                    recordFailure("dxc-dxil-compile-failed");
                    return;
                }
                if (*(reinterpret_cast<uint32_t*>(shader.dxil->GetBufferPointer()) + 1) == 0)
                {
                    recordFailure("dxil-not-signed");
                    return;
                }
#endif

                IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                if (spirv == nullptr)
                {
                    recordFailure("dxc-spirv-compile-failed");
                    return;
                }

                if (!smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(), shader.spirv, smolv::kEncodeFlagStripDebugInfo))
                {
                    spirv->Release();
                    recordFailure("smolv-encode-failed");
                    return;
                }

                spirv->Release();

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
                shader.interpolantMask = recompiler.interpolantMask;
                {
                    struct Variant
                    {
                        ShaderRecompiler::InterpolantVariant kind;
                        IDxcBlob** dxil;
                        std::vector<uint8_t>* spirv;
                        const char* name;
                    };
                    const bool isPixel = recompiler.isPixelShader;
                    const Variant vertexVariants[] = {
                        { ShaderRecompiler::InterpolantVariant::Full, &shader.fullDxil, &shader.fullSpirv, "full" },
                        { ShaderRecompiler::InterpolantVariant::PositionOnly, &shader.posDxil, &shader.posSpirv, "position-only" },
                        { ShaderRecompiler::InterpolantVariant::Velocity, &shader.velDxil, &shader.velSpirv, "velocity" },
                    };
                    // A window-space VS (no float constants, so no WVP) has nothing
                    // to move and gets no velocity variant.
                    const size_t vertexCount = shader.usesFloatConstants ? 3 : 2;
                    const Variant pixelVariants[] = {
                        { ShaderRecompiler::InterpolantVariant::Velocity, &shader.velDxil, &shader.velSpirv, "velocity" },
                    };
                    const Variant* variants = isPixel ? pixelVariants : vertexVariants;
                    const size_t variantCount = isPixel ? 1 : vertexCount;
                    for (size_t vi = 0; vi < variantCount; vi++)
                    {
                        const Variant& v = variants[vi];
                        auto variantFailure = [&](const char* reason)
                        {
                            std::lock_guard<std::mutex> lock(g_failureMutex);
                            g_variantFailures.push_back({hash, fmt::format("{} variant: {}", v.name, reason)});
                            if (*v.dxil != nullptr)
                            {
                                (*v.dxil)->Release();
                                *v.dxil = nullptr;
                            }
                            v.spirv->clear();
                        };
                        recompiler = {};
                        recompiler.interpolantVariant = v.kind;
                        recompiler.recompile(shader.data, include);
                        if (recompiler.specConstantsMask != shader.specConstantsMask ||
                            recompiler.interpolantMask != shader.interpolantMask)
                        {
                            variantFailure("recompiled differently");
                            continue;
                        }
#ifdef XENOS_RECOMP_DXIL
                        *v.dxil = dxcCompiler.compile(recompiler.out, isPixel, shader.specConstantsMask != 0, false);
                        if (*v.dxil == nullptr || *(reinterpret_cast<uint32_t*>((*v.dxil)->GetBufferPointer()) + 1) == 0)
                        {
                            variantFailure("dxc-dxil-compile-failed");
                            continue;
                        }
#endif
                        IDxcBlob* variantSpirv = dxcCompiler.compile(recompiler.out, isPixel, false, true);
                        if (variantSpirv == nullptr)
                        {
                            variantFailure("dxc-spirv-compile-failed");
                            continue;
                        }
                        const bool encoded = smolv::Encode(variantSpirv->GetBufferPointer(), variantSpirv->GetBufferSize(), *v.spirv, smolv::kEncodeFlagStripDebugInfo);
                        variantSpirv->Release();
                        if (!encoded)
                            variantFailure("smolv-encode-failed");
                    }
                }
#endif

                size_t currentProgress = ++progress;
                if ((currentProgress % 10) == 0 || (currentProgress == shaders.size() - 1))
                    fmt::println("Recompiling shaders... {}%", currentProgress / float(shaders.size()) * 100.0f);
            });

        if (!g_failures.empty())
        {
            fmt::println(stderr, "WARNING: {} shader(s) failed to recompile (skipped):", g_failures.size());
            for (const auto& failure : g_failures)
                fmt::println(stderr, "  hash=0x{:016X} reason={}", failure.hash, failure.reason);
            // Remove failed entries so they don't emit zero-size table entries.
            for (const auto& failure : g_failures)
                shaders.erase(failure.hash);
        }

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
        if (!g_variantFailures.empty())
        {
            fmt::println(stderr, "WARNING: {} shader variant(s) failed (the trimmed blob stands in; a failed velocity variant means no motion vector for that shader):", g_variantFailures.size());
            for (const auto& failure : g_variantFailures)
                fmt::println(stderr, "  hash=0x{:016X} {}", failure.hash, failure.reason);
        }
#endif

        fmt::println("Creating shader cache...");

        StringBuffer f;
        f.println("#include \"shader_cache.h\"");
        f.println("ShaderCacheEntry g_shaderCacheEntries[] = {{");

        std::vector<uint8_t> dxil;
        std::vector<uint8_t> spirv;
// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
        std::vector<uint32_t> vertexLayouts;  // flattened (w0, w1) records
#endif

        for (auto& [hash, shader] : shaders)
        {
// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
            // The blobs of a shader sit together: trimmed, then (VS) full and
            // position-only; each slot is (dxil offset, size, spirv offset, size).
            auto append = [&](IDxcBlob* d, const std::vector<uint8_t>& sp, size_t slot[4])
            {
                slot[0] = dxil.size();
                slot[1] = (d != nullptr) ? d->GetBufferSize() : 0;
                if (d != nullptr)
                {
                    dxil.insert(dxil.end(), reinterpret_cast<uint8_t*>(d->GetBufferPointer()),
                        reinterpret_cast<uint8_t*>(d->GetBufferPointer()) + d->GetBufferSize());
                }
                slot[2] = spirv.size();
                slot[3] = sp.size();
                spirv.insert(spirv.end(), sp.begin(), sp.end());
            };
            size_t primary[4], full[4], pos[4], vel[4];
            append(shader.dxil, shader.spirv, primary);
            append(shader.fullDxil, shader.fullSpirv, full);
            append(shader.posDxil, shader.posSpirv, pos);
            append(shader.velDxil, shader.velSpirv, vel);
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 0x{:X}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} }},",
                hash, primary[0], primary[1], primary[2], primary[3], shader.specConstantsMask,
                vertexLayouts.size(), shader.vertexLayout.size(), shader.vfetchCodeOffset,
                shader.usesFloatConstants, shader.interpolantMask,
                full[0], full[1], full[2], full[3], pos[0], pos[1], pos[2], pos[3],
                vel[0], vel[1], vel[2], vel[3]);
            for (const VertexFetchLayoutRecord& r : shader.vertexLayout)
            {
                vertexLayouts.push_back(r.w0);
                vertexLayouts.push_back(r.w1);
            }
#else
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {} }},",
                hash, dxil.size(), (shader.dxil != nullptr) ? shader.dxil->GetBufferSize() : 0, spirv.size(), shader.spirv.size(), shader.specConstantsMask);
#endif

// Reblue specific, most likely needs to be changed for reeot
#ifndef REEOT_RECOMP
            if (shader.dxil != nullptr)
            {
                dxil.insert(dxil.end(), reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()),
                    reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()) + shader.dxil->GetBufferSize());
            }

            spirv.insert(spirv.end(), shader.spirv.begin(), shader.spirv.end());
#endif
        }

        f.println("}};");

// Reblue specific, most likely needs to be changed for reeot
#ifdef REEOT_RECOMP
        f.print("const uint32_t g_shaderVertexLayouts[] = {{");
        for (uint32_t v : vertexLayouts)
            f.print("0x{:X},", v);
        if (vertexLayouts.empty())
            f.print("0");
        f.println("}};");
#endif

        fmt::println("Compressing DXIL cache...");

        int level = ZSTD_maxCLevel();

#ifdef XENOS_RECOMP_DXIL
        std::vector<uint8_t> dxilCompressed(ZSTD_compressBound(dxil.size()));
        dxilCompressed.resize(ZSTD_compress(dxilCompressed.data(), dxilCompressed.size(), dxil.data(), dxil.size(), level));

        f.print("const uint8_t g_compressedDxilCache[] = {{");

        for (auto data : dxilCompressed)
            f.print("{},", data);

        f.println("}};");
        f.println("const size_t g_dxilCacheCompressedSize = {};", dxilCompressed.size());
        f.println("const size_t g_dxilCacheDecompressedSize = {};", dxil.size());
#endif

        fmt::println("Compressing SPIRV cache...");

        std::vector<uint8_t> spirvCompressed(ZSTD_compressBound(spirv.size()));
        spirvCompressed.resize(ZSTD_compress(spirvCompressed.data(), spirvCompressed.size(), spirv.data(), spirv.size(), level));

        f.print("const uint8_t g_compressedSpirvCache[] = {{");

        for (auto data : spirvCompressed)
            f.print("{},", data);

        f.println("}};");

        f.println("const size_t g_spirvCacheCompressedSize = {};", spirvCompressed.size());
        f.println("const size_t g_spirvCacheDecompressedSize = {};", spirv.size());
        f.println("const size_t g_shaderCacheEntryCount = {};", shaders.size());

        writeAllBytes(output, f.out.data(), f.out.size());
    }
    else
    {
        ShaderRecompiler recompiler;
        size_t fileSize;
        recompiler.recompile(readAllBytes(input, fileSize).get(), include);
        writeAllBytes(output, recompiler.out.data(), recompiler.out.size());
    }

    return 0;
}
