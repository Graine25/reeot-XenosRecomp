#include "air_compiler.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fmt/core.h>

// Forwarded to posix_spawn so TOOLCHAINS reaches xcrun: shader_cache.cmake sets
// it to pick a downloaded Metal toolchain, and a null envp would drop it.
extern char** environ;

namespace
{
    // Unlinks on scope exit.
    struct TemporaryPath
    {
        std::string path;

        explicit TemporaryPath(std::string path) : path(std::move(path)) {}
        ~TemporaryPath() { unlink(path.c_str()); }

        TemporaryPath(const TemporaryPath&) = delete;
        TemporaryPath& operator=(const TemporaryPath&) = delete;
    };

    // Runs argv with stdout+stderr sent to logPath. Returns the exit status,
    // or -1 if the child could not be spawned.
    int executeCommand(const char** argv, const std::string& logPath)
    {
        posix_spawn_file_actions_t actions;
        if (posix_spawn_file_actions_init(&actions) != 0)
            return -1;

        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, logPath.c_str(),
            O_WRONLY | O_CREAT | O_TRUNC, 0600);
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

        pid_t pid = 0;
        const int spawned = posix_spawn(&pid, argv[0], &actions, nullptr,
            const_cast<char**>(argv), environ);
        posix_spawn_file_actions_destroy(&actions);

        if (spawned != 0)
            return -1;

        int status = 0;
        if (waitpid(pid, &status, 0) == -1)
            return -1;

        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    std::string readFile(const std::string& path, size_t limit)
    {
        std::ifstream stream(path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        if (text.size() > limit)
            text.resize(limit);
        return text;
    }

    std::string temporaryDirectory()
    {
        if (const char* tmpdir = getenv("TMPDIR"); tmpdir != nullptr && tmpdir[0] != '\0')
        {
            std::string dir(tmpdir);
            if (dir.back() == '/')
                dir.pop_back();
            return dir;
        }
        return "/tmp";
    }
}

std::vector<uint8_t> AirCompiler::compile(const std::string& shaderSource, std::string& error)
{
    // The Metal frontend only reads from disk.
    std::string sourcePathTemplate = temporaryDirectory() + "/xenos_XXXXXX.metal";
    const int sourceFd = mkstemps(sourcePathTemplate.data(), int(strlen(".metal")));
    if (sourceFd == -1)
    {
        error = fmt::format("metal-tempfile-failed: {}", strerror(errno));
        return {};
    }

    const TemporaryPath sourcePath(sourcePathTemplate);
    const TemporaryPath irPath(sourcePath.path + ".air");
    const TemporaryPath metalLibPath(sourcePath.path + ".metallib");
    const TemporaryPath logPath(sourcePath.path + ".log");

    const ssize_t written = write(sourceFd, shaderSource.data(), shaderSource.size());
    close(sourceFd);
    if (written < 0 || size_t(written) != shaderSource.size())
    {
        error = fmt::format("metal-tempfile-write-failed: {}", strerror(errno));
        return {};
    }

    // Debug flags match shaders.cmake's host helpers so both symbolicate alike in
    // frame captures. Math mode stays at the default until we can check it against
    // real output.
    const char* compileCommand[] = {
        "/usr/bin/xcrun", "-sdk", "macosx", "metal",
        "-c", sourcePath.path.c_str(),
        "-o", irPath.path.c_str(),
        "-D__air__",
#ifdef REBLUE_RECOMP
        "-DREBLUE_RECOMP",
#endif
        "-Wno-unused-variable",
        "-frecord-sources", "-gline-tables-only",
        nullptr
    };
    if (const int status = executeCommand(compileCommand, logPath.path); status != 0)
    {
        error = fmt::format("metal-compile-failed(status={}): {}", status,
            readFile(logPath.path, 2048));
        return {};
    }

    const char* linkCommand[] = {
        "/usr/bin/xcrun", "-sdk", "macosx", "metallib",
        irPath.path.c_str(),
        "-o", metalLibPath.path.c_str(),
        nullptr
    };
    if (const int status = executeCommand(linkCommand, logPath.path); status != 0)
    {
        error = fmt::format("metallib-link-failed(status={}): {}", status,
            readFile(logPath.path, 2048));
        return {};
    }

    std::ifstream libStream(metalLibPath.path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(libStream)),
        std::istreambuf_iterator<char>());
    if (data.empty())
        error = "metallib-empty";

    return data;
}
