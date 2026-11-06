#include "shader.h"
#include "shader_recompiler.h"
#include "shader_cache.h"
#include "cache_header_source.h"
#include "dxc_compiler.h"

#include <fstream>
#include <limits>
#include <mutex>

using namespace rengii::gpu;

static std::vector<uint8_t> readAllBytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0)
        throw std::runtime_error("Could not read " + path.string());
    std::vector<uint8_t> data(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size()))
        throw std::runtime_error("Could not read " + path.string());
    return data;
}

static void writeAllBytes(const std::filesystem::path& path, const void* data, size_t size)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    if (!file.write(static_cast<const char*>(data), size))
        throw std::runtime_error("Could not write " + path.string());
    file.close();
    if (!file)
        throw std::runtime_error("Could not finish writing " + path.string());
}

static size_t containerSize(const uint8_t* data, size_t remaining)
{
    if (remaining < sizeof(ShaderContainer))
        return 0;
    const auto& container = *reinterpret_cast<const ShaderContainer*>(data);
    const uint64_t virtualSize = container.virtualSize;
    const uint64_t physicalSize = container.physicalSize;
    const uint64_t shaderOffset = container.shaderOffset;
    const uint64_t descriptionSize = (container.flags & 1) ? sizeof(VertexShader) : sizeof(PixelShader);
    if ((container.flags & 0xFFFFFF00) != 0x102A1100 || container.field1C != 0 || container.field20 != 0 ||
        virtualSize < sizeof(ShaderContainer) || virtualSize + physicalSize > remaining ||
        shaderOffset < sizeof(ShaderContainer) || shaderOffset + descriptionSize > virtualSize ||
        (shaderOffset & 3))
        return 0;
    const auto& shader = *reinterpret_cast<const Shader*>(data + shaderOffset);
    if (uint64_t(shader.physicalOffset) + uint64_t(shader.size) > physicalSize || (shader.size & 3))
        return 0;
    return static_cast<size_t>(virtualSize + physicalSize);
}

struct RecompiledShader
{
    std::vector<uint8_t> data;
    std::vector<uint8_t> spirv;
    std::vector<VertexFetchLayoutRecord> vertexLayout;
    ShaderCacheEntry entry{};
    std::string sourceName;
};

static void recompileDirectory(const std::filesystem::path& input, const std::filesystem::path& output,
    const std::string_view& include, const std::filesystem::path& hlslDump)
{
    std::map<uint64_t, RecompiledShader> shaders;
    std::vector<std::filesystem::path> paths;
    for (const auto& file : std::filesystem::recursive_directory_iterator(input))
        if (file.is_regular_file())
            paths.push_back(file.path());
    std::sort(paths.begin(), paths.end());

    size_t containerCount = 0;
    for (const auto& path : paths)
    {
        const auto data = readAllBytes(path);
        for (size_t offset = 0; offset + sizeof(ShaderContainer) <= data.size();)
        {
            const size_t size = containerSize(data.data() + offset, data.size() - offset);
            if (!size)
            {
                offset += sizeof(uint32_t);
                continue;
            }
            ++containerCount;
            const uint64_t hash = XXH3_64bits(data.data() + offset, size);
            auto [it, inserted] = shaders.try_emplace(hash);
            if (inserted)
            {
                it->second.data.assign(data.begin() + offset, data.begin() + offset + size);
                it->second.sourceName = path.stem().string();
            }
            else if (it->second.data.size() != size || std::memcmp(it->second.data.data(), data.data() + offset, size))
                throw std::runtime_error(fmt::format("Container hash collision: {:016X}", hash));
            offset += size;
        }
    }
    if (shaders.empty())
        throw std::runtime_error("No shader containers found");
    fmt::println("Found {} containers, {} unique shaders", containerCount, shaders.size());
    if (!hlslDump.empty())
        std::filesystem::create_directories(hlslDump);

    std::vector<decltype(shaders)::value_type*> jobs;
    for (auto& shader : shaders)
        jobs.push_back(&shader);
    std::atomic<size_t> next = 0;
    std::atomic<size_t> completed = 0;
    std::mutex mutex;
    size_t failures = 0;
    auto worker = [&]
    {
        for (size_t index; (index = next++) < jobs.size();)
        {
            auto& [hash, shader] = *jobs[index];
            try
            {
                ShaderRecompiler recompiler;
                recompiler.recompile(shader.data.data(), include);
                if (recompiler.unsupportedExports)
                    throw std::runtime_error(fmt::format("Unsupported exports: {:X}", recompiler.unsupportedExports));
                if (!hlslDump.empty())
                {
                    const auto path = hlslDump / fmt::format("{:016X}_{}.hlsl", hash, shader.sourceName);
                    writeAllBytes(path, recompiler.out.data(), recompiler.out.size());
                }
                thread_local DxcCompiler compiler;
                IDxcBlob* spirv = compiler.compile(recompiler.out, recompiler.isPixelShader);
                const bool encoded = smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(),
                    shader.spirv, smolv::kEncodeFlagStripDebugInfo);
                spirv->Release();
                if (!encoded)
                    throw std::runtime_error("SMOL-V encoding failed");

                shader.vertexLayout = std::move(recompiler.vertexLayout);
                auto& entry = shader.entry;
                entry.hash = hash;
                entry.stage = recompiler.isPixelShader ? ShaderStage::Pixel : ShaderStage::Vertex;
                entry.specConstantsMask = recompiler.specConstantsMask;
                entry.physicalCodeOffset = recompiler.physicalCodeOffset;
                entry.floatConstantRegs = recompiler.floatConstantRegs;
                entry.interpolantMask = recompiler.interpolantMask;
                entry.textureSlotMask = recompiler.textureSlotMask;
                entry.pointSlotMask = recompiler.pointSlotMask;
                entry.linearSlotMask = recompiler.linearSlotMask;
                entry.mipPointSlotMask = recompiler.mipPointSlotMask;
                entry.mipLinearSlotMask = recompiler.mipLinearSlotMask;
                entry.mipBaseSlotMask = recompiler.mipBaseSlotMask;
                entry.anisoOffSlotMask = recompiler.anisoOffSlotMask;
                entry.usesMemExport = recompiler.usesMemExport;
                entry.coverageOutput = recompiler.coverageOutput;
                const size_t count = ++completed;
                if (count % 64 == 0 || count == jobs.size())
                {
                    std::lock_guard lock(mutex);
                    fmt::println("Recompiled {}/{} shaders", count, jobs.size());
                }
            }
            catch (const std::exception& error)
            {
                std::lock_guard lock(mutex);
                ++failures;
                fmt::println(stderr, "Shader {:016X} ({}): {}", hash, shader.sourceName, error.what());
            }
        }
    };
    std::vector<std::thread> workers;
    for (size_t i = 0; i < std::min<size_t>(4, jobs.size()); ++i)
        workers.emplace_back(worker);
    for (auto& thread : workers)
        thread.join();
    if (failures)
        throw std::runtime_error(fmt::format("{} shaders failed; cache was not written", failures));

    std::vector<uint8_t> spirv;
    std::vector<VertexFetchLayoutRecord> vertexLayout;
    StringBuffer source;
    source.println("#include \"shader_cache.h\"\n\nnamespace rengii::gpu\n{{");
    source.println("const ShaderCacheEntry g_shaderCacheEntries[] = {{");
    for (auto& [hash, shader] : shaders)
    {
        if (shader.spirv.size() + spirv.size() > std::numeric_limits<uint32_t>::max() ||
            shader.vertexLayout.size() + vertexLayout.size() > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Shader cache exceeds 32-bit offsets");
        auto& entry = shader.entry;
        entry.spirvOffset = uint32_t(spirv.size());
        entry.spirvSize = uint32_t(shader.spirv.size());
        entry.vertexFetchOffset = uint32_t(vertexLayout.size());
        entry.vertexFetchCount = uint32_t(shader.vertexLayout.size());
        source.println("    {{0x{:016X}ull, {}, {}, ShaderStage::{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}},",
            hash, entry.spirvOffset, entry.spirvSize, entry.stage == ShaderStage::Pixel ? "Pixel" : "Vertex",
            entry.specConstantsMask, entry.vertexFetchOffset, entry.vertexFetchCount, entry.physicalCodeOffset,
            entry.floatConstantRegs, entry.interpolantMask, entry.textureSlotMask, entry.pointSlotMask,
            entry.linearSlotMask, entry.mipPointSlotMask, entry.mipLinearSlotMask, entry.mipBaseSlotMask,
            entry.anisoOffSlotMask, entry.usesMemExport, entry.coverageOutput);
        spirv.insert(spirv.end(), shader.spirv.begin(), shader.spirv.end());
        vertexLayout.insert(vertexLayout.end(), shader.vertexLayout.begin(), shader.vertexLayout.end());
    }
    source.println("}};\nconst size_t g_shaderCacheEntryCount = {};", shaders.size());
    source.println("const ShaderVertexFetch g_shaderVertexFetches[] = {{");
    if (vertexLayout.empty())
        source.println("    {{0, 0}},");
    for (const auto& fetch : vertexLayout)
        source.println("    {{{}, {}}},", fetch.w0, fetch.w1);
    source.println("}};\nconst size_t g_shaderVertexFetchCount = {};", vertexLayout.size());

    std::vector<uint8_t> compressed(ZSTD_compressBound(spirv.size()));
    const size_t size = ZSTD_compress(compressed.data(), compressed.size(), spirv.data(), spirv.size(), ZSTD_maxCLevel());
    if (ZSTD_isError(size))
        throw std::runtime_error(ZSTD_getErrorName(size));
    compressed.resize(size);
    source.print("const uint8_t g_compressedSpirvCache[] = {{");
    for (size_t i = 0; i < compressed.size(); ++i)
    {
        if (i % 24 == 0)
            source.print("\n    ");
        source.print("{},", compressed[i]);
    }
    source.println("\n}};\nconst size_t g_spirvCacheCompressedSize = {};", compressed.size());
    source.println("const size_t g_spirvCacheDecompressedSize = {};\n}}", spirv.size());
    writeAllBytes(output.parent_path() / "shader_cache.h", kShaderCacheHeader, sizeof(kShaderCacheHeader) - 1);
    auto binary = output;
    binary.replace_extension(".bin");
    writeAllBytes(binary, compressed.data(), compressed.size());
    writeAllBytes(output, source.out.data(), source.out.size());
    fmt::println("Wrote {} shaders, {} vertex fetches, {} compressed bytes to {}", shaders.size(),
        vertexLayout.size(), compressed.size(), output.string());
}

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        fmt::println(stderr, "Usage: XenosRecomp INPUT OUTPUT SHADER_COMMON [HLSL_DIR]");
        return 1;
    }
    try
    {
        if (std::filesystem::exists(argv[2]) && std::filesystem::equivalent(argv[1], argv[2]))
            throw std::runtime_error("Input and output must differ");
        const auto includeData = readAllBytes(argv[3]);
        const std::string_view include(reinterpret_cast<const char*>(includeData.data()), includeData.size());
        if (std::filesystem::is_directory(argv[1]))
        {
            recompileDirectory(argv[1], argv[2], include, argc > 4 ? argv[4] : "");
        }
        else
        {
            const auto data = readAllBytes(argv[1]);
            if (!containerSize(data.data(), data.size()))
                throw std::runtime_error("Invalid shader container");
            ShaderRecompiler recompiler;
            recompiler.recompile(data.data(), include);
            if (recompiler.unsupportedExports)
                throw std::runtime_error("Shader contains unsupported exports");
            writeAllBytes(argv[2], recompiler.out.data(), recompiler.out.size());
        }
    }
    catch (const std::exception& error)
    {
        fmt::println(stderr, "XenosRecomp: {}", error.what());
        return 1;
    }
    return 0;
}
