project(XenosRecomp)

find_package(Threads REQUIRED)

set(SMOLV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/smol-v/source")

add_executable(XenosRecomp 
    constant_table.h
    dxc_compiler.cpp
    dxc_compiler.h
    main.cpp
    pch.h
    shader.h
    shader_cache.h
    shader_code.h
    shader_recompiler.cpp
    shader_recompiler.h
    "${SMOLV_SOURCE_DIR}/smolv.cpp")

target_link_libraries(XenosRecomp PRIVATE
    Threads::Threads
    Microsoft::DirectXShaderCompiler
    xxHash::xxhash
    libzstd_static
    fmt::fmt)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader_cache.h")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shader_cache.h" SHADER_CACHE_HEADER)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cache_header_source.h"
    "inline constexpr char kShaderCacheHeader[] = R\"CACHE(${SHADER_CACHE_HEADER})CACHE\";\n")

target_include_directories(XenosRecomp PRIVATE ${SMOLV_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(XenosRecomp PRIVATE RENGII_RECOMP=1)

target_precompile_headers(XenosRecomp PRIVATE pch.h)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(XenosRecomp PRIVATE -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions)
endif()

if (WIN32)
    target_compile_definitions(XenosRecomp PRIVATE _CRT_SECURE_NO_WARNINGS)
    add_custom_command(TARGET XenosRecomp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:XenosRecomp> $<TARGET_FILE_DIR:XenosRecomp>
        COMMAND_EXPAND_LISTS
    )
endif()
