cmake_minimum_required(VERSION 3.20)

project(rnnoise C CXX)

option(RNNOISE_BUILD_EXAMPLES "Build example programs" ON)
option(RNNOISE_ENABLE_WINML "Use Windows ML for the RNN backend" ON)
option(RNNOISE_ENABLE_ONNX "Use ONNX Runtime for the RNN backend" OFF)
option(RNNOISE_COMPARE_RNN_BACKENDS "Run native CPU and external RNN backends together and compare outputs" OFF)
set(RNNOISE_WINML_PACKAGE_ROOT "" CACHE PATH "Path to the Microsoft.Windows.AI.MachineLearning NuGet package root")
set(RNNOISE_ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX Runtime native package root")
set(RNNOISE_ONNX_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/rnnoise_rnn.onnx" CACHE FILEPATH "ONNX model deployed next to example executables")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_GENERATOR_PLATFORM)
  set(_RNNOISE_PLATFORM "${CMAKE_GENERATOR_PLATFORM}")
elseif(CMAKE_VS_PLATFORM_NAME)
  set(_RNNOISE_PLATFORM "${CMAKE_VS_PLATFORM_NAME}")
elseif(CMAKE_SYSTEM_PROCESSOR)
  set(_RNNOISE_PLATFORM "${CMAKE_SYSTEM_PROCESSOR}")
else()
  set(_RNNOISE_PLATFORM "x64")
endif()
string(TOUPPER "${_RNNOISE_PLATFORM}" _RNNOISE_PLATFORM_UPPER)
if(_RNNOISE_PLATFORM_UPPER MATCHES "^(AMD64|X64|X86_64)$")
  set(RNNOISE_WINDOWS_RID "win-x64")
  set(RNNOISE_NATIVE_ARCH "x64")
elseif(_RNNOISE_PLATFORM_UPPER MATCHES "^ARM64EC$")
  set(RNNOISE_WINDOWS_RID "win-arm64ec")
  set(RNNOISE_NATIVE_ARCH "arm64ec")
elseif(_RNNOISE_PLATFORM_UPPER MATCHES "^(ARM64|AARCH64)$")
  set(RNNOISE_WINDOWS_RID "win-arm64")
  set(RNNOISE_NATIVE_ARCH "arm64")
else()
  message(FATAL_ERROR "Unsupported rnnoise Windows architecture: ${_RNNOISE_PLATFORM}")
endif()

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
  set(RNNOISE_WINML_NATIVE_DIR "${RNNOISE_WINML_PACKAGE_ROOT}/runtimes/${RNNOISE_WINDOWS_RID}/native")
  set(RNNOISE_WINML_SHARED_PROVIDERS_DLL "")
  foreach(_RNNOISE_SHARED_PROVIDER_CANDIDATE
      "${RNNOISE_WINML_NATIVE_DIR}/onnxruntime_providers_shared.dll"
      "$ENV{USERPROFILE}/.nuget/packages/microsoft.windowsappsdk.ml/1.8.2091/runtimes-framework/${RNNOISE_WINDOWS_RID}/native/onnxruntime_providers_shared.dll")
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
      file(GLOB _RNNOISE_ORT_FOUNDRY_PACKAGES
        LIST_DIRECTORIES true
        "$ENV{USERPROFILE}/.nuget/packages/microsoft.ml.onnxruntime.foundry/*")
      if(_RNNOISE_ORT_PACKAGES)
        list(SORT _RNNOISE_ORT_PACKAGES COMPARE NATURAL ORDER DESCENDING)
        list(GET _RNNOISE_ORT_PACKAGES 0 RNNOISE_ONNXRUNTIME_ROOT)
      elseif(_RNNOISE_ORT_FOUNDRY_PACKAGES)
        list(SORT _RNNOISE_ORT_FOUNDRY_PACKAGES COMPARE NATURAL ORDER DESCENDING)
        list(GET _RNNOISE_ORT_FOUNDRY_PACKAGES 0 RNNOISE_ONNXRUNTIME_ROOT)
      else()
        message(FATAL_ERROR "RNNOISE_ENABLE_ONNX requires ONNX Runtime. Set RNNOISE_ONNXRUNTIME_ROOT, vendor third_party/onnxruntime_nuget, or install Microsoft.ML.OnnxRuntime.")
      endif()
    endif()
  endif()
  set(ONNXRUNTIME_ROOT "${RNNOISE_ONNXRUNTIME_ROOT}")
  if(EXISTS "${ONNXRUNTIME_ROOT}/build/native/include/onnxruntime_c_api.h")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/build/native/include")
  elseif(EXISTS "${ONNXRUNTIME_ROOT}/include/winml/onnxruntime_c_api.h")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/winml")
  elseif(EXISTS "${ONNXRUNTIME_ROOT}/build/onnxruntime.targets")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/build/native/include")
  else()
    message(FATAL_ERROR "Could not locate onnxruntime_c_api.h under ${ONNXRUNTIME_ROOT}")
  endif()
  if(EXISTS "${ONNXRUNTIME_ROOT}/runtimes/${RNNOISE_WINDOWS_RID}/native/onnxruntime.lib")
    set(ONNXRUNTIME_NATIVE_DIR "${ONNXRUNTIME_ROOT}/runtimes/${RNNOISE_WINDOWS_RID}/native")
  elseif(EXISTS "${ONNXRUNTIME_ROOT}/lib/native/${RNNOISE_NATIVE_ARCH}/onnxruntime.lib")
    set(ONNXRUNTIME_NATIVE_DIR "${ONNXRUNTIME_ROOT}/lib/native/${RNNOISE_NATIVE_ARCH}")
  else()
    message(FATAL_ERROR "Could not locate onnxruntime.lib under ${ONNXRUNTIME_ROOT}")
  endif()
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

function(rnnoise_deploy_onnx_model target)
  if(RNNOISE_ENABLE_WINML OR RNNOISE_ENABLE_ONNX)
    if(NOT EXISTS "${RNNOISE_ONNX_MODEL}")
      message(FATAL_ERROR "RNNOISE_ONNX_MODEL does not exist: ${RNNOISE_ONNX_MODEL}")
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:${target}>/models"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${RNNOISE_ONNX_MODEL}"
        "$<TARGET_FILE_DIR:${target}>/models/rnnoise_rnn.onnx"
    )
  endif()
endfunction()

if(RNNOISE_BUILD_EXAMPLES)
  add_executable(rnnoise_demo examples/rnnoise_demo.c)
  target_link_libraries(rnnoise_demo PRIVATE rnnoise)
  target_compile_definitions(rnnoise_demo PRIVATE _CRT_SECURE_NO_WARNINGS)
  rnnoise_deploy_onnx_model(rnnoise_demo)
  add_executable(rnnoise_compare_random tools/rnnoise_compare_random.c)
  target_link_libraries(rnnoise_compare_random PRIVATE rnnoise)
  target_compile_definitions(rnnoise_compare_random PRIVATE _CRT_SECURE_NO_WARNINGS)
  rnnoise_deploy_onnx_model(rnnoise_compare_random)
  if(RNNOISE_ENABLE_WINML)
    add_executable(rnnoise_winml_ep_tool tools/rnnoise_winml_ep_tool.cpp)
    target_link_libraries(rnnoise_winml_ep_tool PRIVATE rnnoise)
    target_include_directories(rnnoise_winml_ep_tool PRIVATE src)
    target_compile_features(rnnoise_winml_ep_tool PRIVATE cxx_std_17)
    rnnoise_deploy_onnx_model(rnnoise_winml_ep_tool)

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
    add_custom_command(TARGET rnnoise_compare_random POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_NATIVE_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:rnnoise_compare_random>/onnxruntime.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_NATIVE_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:rnnoise_compare_random>/onnxruntime_providers_shared.dll"
    )
  endif()
endif()
