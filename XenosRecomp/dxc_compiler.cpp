#include "dxc_compiler.h"

DxcCompiler::DxcCompiler()
{
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler))))
        throw std::runtime_error("Could not create DXC compiler");
}

DxcCompiler::~DxcCompiler()
{
    dxcCompiler->Release();
}

IDxcBlob* DxcCompiler::compile(const std::string& shaderSource, bool pixelShader)
{
    DxcBuffer source{};
    source.Ptr = shaderSource.data();
    source.Size = shaderSource.size();
    source.Encoding = DXC_CP_UTF8;

    std::vector<const wchar_t*> args = {
        L"-T", pixelShader ? L"ps_6_6" : L"vs_6_6",
        L"-E", L"shaderMain", L"-HV", L"2021",
        L"-spirv", L"-fspv-target-env=vulkan1.2", L"-fvk-use-dx-layout",
        L"-DRENGII_RECOMP=1", L"-Qstrip_debug"
    };
    if (!pixelShader)
        args.push_back(L"-fvk-invert-y");

    IDxcResult* result = nullptr;
    const HRESULT hr = dxcCompiler->Compile(&source, args.data(), uint32_t(args.size()), nullptr, IID_PPV_ARGS(&result));
    if (FAILED(hr) || !result)
        throw std::runtime_error("DXC invocation failed");

    IDxcBlob* object = nullptr;
    HRESULT status = E_FAIL;
    result->GetStatus(&status);
    if (SUCCEEDED(status))
        result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr);
    else
    {
        IDxcBlobUtf8* errors = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        std::string message = errors ? errors->GetStringPointer() : "DXC compilation failed";
        if (errors)
            errors->Release();
        result->Release();
        throw std::runtime_error(message);
    }
    result->Release();
    if (!object)
        throw std::runtime_error("DXC returned no shader");
    return object;
}
