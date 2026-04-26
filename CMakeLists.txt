cmake_minimum_required(VERSION 3.16)

if(APPLE AND NOT CMAKE_OSX_SYSROOT)
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE INFEREDGE_MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(INFEREDGE_MACOS_SDK_PATH)
        set(CMAKE_OSX_SYSROOT "${INFEREDGE_MACOS_SDK_PATH}" CACHE PATH "macOS SDK path" FORCE)
    endif()
endif()

project(inferedge_runtime VERSION 0.1.0 LANGUAGES CXX)

option(INFEREDGE_ENABLE_ORT "Enable ONNX Runtime backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(inferedge-runtime
    src/main.cpp
    src/cli.cpp
    src/engine.cpp
    src/engines/onnxruntime_engine.cpp
)

target_include_directories(inferedge-runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(APPLE AND CMAKE_OSX_SYSROOT)
    target_include_directories(inferedge-runtime
        SYSTEM PRIVATE
            ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1
    )
endif()

if(INFEREDGE_ENABLE_ORT)
    target_compile_definitions(inferedge-runtime PRIVATE INFEREDGE_ENABLE_ORT=1)
endif()
