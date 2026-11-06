#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;

    DxcCompiler();
    ~DxcCompiler();
    DxcCompiler(const DxcCompiler&) = delete;
    DxcCompiler& operator=(const DxcCompiler&) = delete;

    IDxcBlob* compile(const std::string& shaderSource, bool pixelShader);
};
