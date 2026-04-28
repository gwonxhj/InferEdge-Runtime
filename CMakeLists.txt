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
set(INFEREDGE_ORT_ROOT "" CACHE PATH "Path to ONNX Runtime C/C++ package root")
option(INFEREDGE_ENABLE_TENSORRT "Enable TensorRT backend link validation" OFF)
set(INFEREDGE_TENSORRT_ROOT "" CACHE PATH "Optional TensorRT root path")
set(INFEREDGE_CUDA_ROOT "" CACHE PATH "Optional CUDA root path")
option(INFEREDGE_ENABLE_OPENCV "Enable OpenCV image input preprocessing" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(inferedge-runtime
    src/main.cpp
    src/cli.cpp
    src/engine.cpp
    src/lab_worker_request.cpp
    src/manifest.cpp
    src/result_writer.cpp
    src/engines/onnxruntime_engine.cpp
    src/engines/tensorrt_engine.cpp
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
    if(NOT INFEREDGE_ORT_ROOT)
        message(FATAL_ERROR "INFEREDGE_ENABLE_ORT=ON requires -DINFEREDGE_ORT_ROOT=/path/to/onnxruntime")
    endif()

    set(INFEREDGE_ORT_INCLUDE "${INFEREDGE_ORT_ROOT}/include")
    set(INFEREDGE_ORT_LIBRARY_DIR "${INFEREDGE_ORT_ROOT}/lib")

    if(NOT EXISTS "${INFEREDGE_ORT_INCLUDE}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime C++ header not found: ${INFEREDGE_ORT_INCLUDE}/onnxruntime_cxx_api.h")
    endif()

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${INFEREDGE_ORT_LIBRARY_DIR}"
        NO_DEFAULT_PATH
    )

    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime library not found under: ${INFEREDGE_ORT_LIBRARY_DIR}")
    endif()

    target_include_directories(inferedge-runtime
        SYSTEM PRIVATE
            "${INFEREDGE_ORT_INCLUDE}"
    )
    target_link_libraries(inferedge-runtime PRIVATE "${ONNXRUNTIME_LIBRARY}")
    target_compile_definitions(inferedge-runtime PRIVATE INFEREDGE_ENABLE_ORT=1 INFEREDGE_ORT_LINKED=1)
    set_property(TARGET inferedge-runtime APPEND PROPERTY BUILD_RPATH "${INFEREDGE_ORT_LIBRARY_DIR}")
endif()

if(INFEREDGE_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
    target_link_libraries(inferedge-runtime PRIVATE ${OpenCV_LIBS})
    target_include_directories(inferedge-runtime SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(inferedge-runtime PRIVATE INFEREDGE_ENABLE_OPENCV=1 INFEREDGE_OPENCV_LINKED=1)
endif()

if(INFEREDGE_ENABLE_TENSORRT)
    set(INFEREDGE_TENSORRT_INCLUDE_CANDIDATES
        /usr/include/aarch64-linux-gnu
    )
    set(INFEREDGE_TENSORRT_LIBRARY_CANDIDATES
        /usr/lib/aarch64-linux-gnu
    )

    if(INFEREDGE_TENSORRT_ROOT)
        list(APPEND INFEREDGE_TENSORRT_INCLUDE_CANDIDATES
            "${INFEREDGE_TENSORRT_ROOT}/include"
            "${INFEREDGE_TENSORRT_ROOT}/include/aarch64-linux-gnu"
        )
        list(APPEND INFEREDGE_TENSORRT_LIBRARY_CANDIDATES
            "${INFEREDGE_TENSORRT_ROOT}/lib"
            "${INFEREDGE_TENSORRT_ROOT}/lib/aarch64-linux-gnu"
            "${INFEREDGE_TENSORRT_ROOT}/lib64"
        )
    endif()

    set(INFEREDGE_CUDA_INCLUDE_CANDIDATES
        /usr/local/cuda/include
    )
    set(INFEREDGE_CUDA_LIBRARY_CANDIDATES
        /usr/local/cuda/lib64
    )

    if(INFEREDGE_CUDA_ROOT)
        list(APPEND INFEREDGE_CUDA_INCLUDE_CANDIDATES
            "${INFEREDGE_CUDA_ROOT}/include"
        )
        list(APPEND INFEREDGE_CUDA_LIBRARY_CANDIDATES
            "${INFEREDGE_CUDA_ROOT}/lib64"
            "${INFEREDGE_CUDA_ROOT}/lib"
        )
    endif()

    find_path(TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS ${INFEREDGE_TENSORRT_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(NOT TENSORRT_INCLUDE_DIR)
        message(FATAL_ERROR "TensorRT header NvInfer.h not found. Set -DINFEREDGE_TENSORRT_ROOT=/path/to/TensorRT or build on Jetson.")
    endif()

    find_library(TENSORRT_LIBRARY
        NAMES nvinfer
        PATHS ${INFEREDGE_TENSORRT_LIBRARY_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT library nvinfer not found.")
    endif()

    find_path(CUDA_RUNTIME_INCLUDE_DIR
        NAMES cuda_runtime_api.h cuda_runtime.h
        PATHS ${INFEREDGE_CUDA_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(NOT CUDA_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "CUDA runtime header not found. Set -DINFEREDGE_CUDA_ROOT=/usr/local/cuda.")
    endif()

    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        PATHS ${INFEREDGE_CUDA_LIBRARY_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(NOT CUDA_RUNTIME_LIBRARY)
        message(FATAL_ERROR "CUDA runtime library cudart not found.")
    endif()

    target_include_directories(inferedge-runtime
        SYSTEM PRIVATE
            "${TENSORRT_INCLUDE_DIR}"
            "${CUDA_RUNTIME_INCLUDE_DIR}"
    )
    target_link_libraries(inferedge-runtime PRIVATE "${TENSORRT_LIBRARY}" "${CUDA_RUNTIME_LIBRARY}")
    target_compile_definitions(inferedge-runtime PRIVATE INFEREDGE_ENABLE_TENSORRT=1 INFEREDGE_TENSORRT_LINKED=1)

    get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
    get_filename_component(CUDA_RUNTIME_LIBRARY_DIR "${CUDA_RUNTIME_LIBRARY}" DIRECTORY)
    set_property(TARGET inferedge-runtime APPEND PROPERTY BUILD_RPATH "${TENSORRT_LIBRARY_DIR}" "${CUDA_RUNTIME_LIBRARY_DIR}")
endif()
