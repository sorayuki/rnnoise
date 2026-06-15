cmake_minimum_required(VERSION 3.20)

project(rnnoise C CXX)

option(RNNOISE_BUILD_EXAMPLES "Build example programs" ON)
option(RNNOISE_ENABLE_WINML "Use Windows ML for the RNN backend" ON)
option(RNNOISE_ENABLE_ONNX "Use ONNX Runtime for the RNN backend" OFF)
option(RNNOISE_COMPARE_RNN_BACKENDS "Run native CPU and external RNN backends together and compare outputs" OFF)
set(RNNOISE_WINML_PACKAGE_ROOT "" CACHE PATH "Path to the Microsoft.Windows.AI.MachineLearning NuGet package root")
set(RNNOISE_ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX Runtime native package root")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(rnnoise STATIC
  src/denoise.c
  src/rnn.c
  src/pitch.c
  src/kiss_fft.c
  src/celt_lpc.c
  src/nnet.c
  src/nnet_default.c
  src/parse_lpcnet_weights.c
  src/rnnoise_data.c
  src/rnnoise_tables.c
  src/rnn_ort_common.cpp
)

if(RNNOISE_ENABLE_WINML)
  if(NOT RNNOISE_WINML_PACKAGE_ROOT)
    file(GLOB _RNNOISE_WINML_PACKAGES
      LIST_DIRECTORIES true
      "$ENV{USERPROFILE}/.nuget/packages/microsoft.windows.ai.machinelearning/*")
    if(NOT _RNNOISE_WINML_PACKAGES)
      message(FATAL_ERROR "RNNOISE_ENABLE_WINML requires Microsoft.Windows.AI.MachineLearning. Set RNNOISE_WINML_PACKAGE_ROOT to the NuGet package root.")
    endif()
    list(SORT _RNNOISE_WINML_PACKAGES COMPARE NATURAL ORDER DESCENDING)
    list(GET _RNNOISE_WINML_PACKAGES 0 RNNOISE_WINML_PACKAGE_ROOT)
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${RNNOISE_WINML_PACKAGE_ROOT}/build/cmake")
  find_package(microsoft.windows.ai.machinelearning CONFIG REQUIRED)
  set(RNNOISE_WINML_NATIVE_DIR "${RNNOISE_WINML_PACKAGE_ROOT}/runtimes/win-x64/native")
  set(RNNOISE_WINML_SHARED_PROVIDERS_DLL "")
  foreach(_RNNOISE_SHARED_PROVIDER_CANDIDATE
      "${RNNOISE_WINML_NATIVE_DIR}/onnxruntime_providers_shared.dll"
      "$ENV{USERPROFILE}/.nuget/packages/microsoft.windowsappsdk.ml/1.8.2091/runtimes-framework/win-x64/native/onnxruntime_providers_shared.dll")
    if(EXISTS "${_RNNOISE_SHARED_PROVIDER_CANDIDATE}")
      set(RNNOISE_WINML_SHARED_PROVIDERS_DLL "${_RNNOISE_SHARED_PROVIDER_CANDIDATE}")
      break()
    endif()
  endforeach()
  target_sources(rnnoise PRIVATE src/rnn_winml.cpp)
  target_compile_definitions(rnnoise PRIVATE RNNOISE_ENABLE_WINML)
  target_compile_features(rnnoise PRIVATE cxx_std_17)
  target_link_libraries(rnnoise PUBLIC WindowsML::Api WindowsML::OnnxRuntime)
elseif(RNNOISE_ENABLE_ONNX)
  if(NOT RNNOISE_ONNXRUNTIME_ROOT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime_nuget/build/native/include/onnxruntime_c_api.h")
      set(RNNOISE_ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime_nuget")
    else()
      file(GLOB _RNNOISE_ORT_PACKAGES
        LIST_DIRECTORIES true
        "$ENV{USERPROFILE}/.nuget/packages/microsoft.ml.onnxruntime/*")
      if(_RNNOISE_ORT_PACKAGES)
        list(SORT _RNNOISE_ORT_PACKAGES COMPARE NATURAL ORDER DESCENDING)
        list(GET _RNNOISE_ORT_PACKAGES 0 RNNOISE_ONNXRUNTIME_ROOT)
      else()
        message(FATAL_ERROR "RNNOISE_ENABLE_ONNX requires ONNX Runtime. Set RNNOISE_ONNXRUNTIME_ROOT, vendor third_party/onnxruntime_nuget, or install Microsoft.ML.OnnxRuntime.")
      endif()
    endif()
  endif()
  set(ONNXRUNTIME_ROOT "${RNNOISE_ONNXRUNTIME_ROOT}")
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/build/native/include")
  set(ONNXRUNTIME_NATIVE_DIR "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
  target_sources(rnnoise PRIVATE src/rnn_onnx.cpp)
  target_include_directories(rnnoise PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
  target_compile_definitions(rnnoise PRIVATE RNNOISE_ENABLE_ONNX)
  target_compile_features(rnnoise PRIVATE cxx_std_17)
  target_link_libraries(rnnoise PUBLIC "${ONNXRUNTIME_NATIVE_DIR}/onnxruntime.lib")
endif()

if(RNNOISE_COMPARE_RNN_BACKENDS)
  if(NOT RNNOISE_ENABLE_WINML AND NOT RNNOISE_ENABLE_ONNX)
    message(FATAL_ERROR "RNNOISE_COMPARE_RNN_BACKENDS requires RNNOISE_ENABLE_WINML or RNNOISE_ENABLE_ONNX")
  endif()
  target_compile_definitions(rnnoise PRIVATE RNNOISE_COMPARE_RNN_BACKENDS)
endif()

target_include_directories(rnnoise
  PUBLIC include
  PRIVATE src
)

if(RNNOISE_ENABLE_ONNX)
  target_include_directories(rnnoise PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
endif()

target_compile_definitions(rnnoise
  PRIVATE
    RNNOISE_BUILD
    _CRT_SECURE_NO_WARNINGS
)

if(MSVC)
  target_compile_options(rnnoise PRIVATE /W3)
else()
  target_compile_options(rnnoise PRIVATE -Wall -Wextra -Wno-sign-compare)
  target_link_libraries(rnnoise PUBLIC m)
endif()

if(RNNOISE_BUILD_EXAMPLES)
  add_executable(rnnoise_demo examples/rnnoise_demo.c)
  target_link_libraries(rnnoise_demo PRIVATE rnnoise)
  target_compile_definitions(rnnoise_demo PRIVATE _CRT_SECURE_NO_WARNINGS)
  if(RNNOISE_ENABLE_WINML)
    add_executable(rnnoise_winml_ep_tool tools/rnnoise_winml_ep_tool.cpp)
    target_link_libraries(rnnoise_winml_ep_tool PRIVATE rnnoise)
    target_include_directories(rnnoise_winml_ep_tool PRIVATE src)
    target_compile_features(rnnoise_winml_ep_tool PRIVATE cxx_std_17)

    add_custom_command(TARGET rnnoise_demo POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:WindowsML::Api>"
        "$<TARGET_FILE_DIR:rnnoise_demo>/Microsoft.Windows.AI.MachineLearning.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:WindowsML::OnnxRuntime>"
        "$<TARGET_FILE_DIR:rnnoise_demo>/onnxruntime.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${WINML_DIRECTML_DLL}"
        "$<TARGET_FILE_DIR:rnnoise_demo>/DirectML.dll"
    )
    if(RNNOISE_WINML_SHARED_PROVIDERS_DLL)
      add_custom_command(TARGET rnnoise_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${RNNOISE_WINML_SHARED_PROVIDERS_DLL}"
          "$<TARGET_FILE_DIR:rnnoise_demo>/onnxruntime_providers_shared.dll"
      )
    endif()
    add_custom_command(TARGET rnnoise_winml_ep_tool POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:WindowsML::Api>"
        "$<TARGET_FILE_DIR:rnnoise_winml_ep_tool>/Microsoft.Windows.AI.MachineLearning.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:WindowsML::OnnxRuntime>"
        "$<TARGET_FILE_DIR:rnnoise_winml_ep_tool>/onnxruntime.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${WINML_DIRECTML_DLL}"
        "$<TARGET_FILE_DIR:rnnoise_winml_ep_tool>/DirectML.dll"
    )
    if(RNNOISE_WINML_SHARED_PROVIDERS_DLL)
      add_custom_command(TARGET rnnoise_winml_ep_tool POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${RNNOISE_WINML_SHARED_PROVIDERS_DLL}"
          "$<TARGET_FILE_DIR:rnnoise_winml_ep_tool>/onnxruntime_providers_shared.dll"
      )
    endif()
  elseif(RNNOISE_ENABLE_ONNX)
    add_custom_command(TARGET rnnoise_demo POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_NATIVE_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:rnnoise_demo>/onnxruntime.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_NATIVE_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:rnnoise_demo>/onnxruntime_providers_shared.dll"
    )
  endif()
endif()
