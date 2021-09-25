#=======================================
#                PROJECT
#=======================================

cmake_minimum_required(VERSION 3.8)
project("CUDANGINE" CXX CUDA)

#=======================================
#               LIBRARIES
#=======================================

find_package("CUDA" REQUIRED)

#=======================================
#                LIBRARY                
#=======================================

set(cudangine_src
    "cudangine/cudangine/exception.cu")

set(cudangine_header
    "cudangine/cudangine/buffer.hpp"
    "cudangine/cudangine/buffer.hxx"
    "cudangine/cudangine/exception.hpp")

cuda_add_library(cudangine  ${cudangine_src} ${cudangine_header})
target_include_directories(cudangine PUBLIC "cudangine")
cuda_include_directories("cudangine")

target_compile_features(cudangine PUBLIC cxx_noexcept)
target_compile_features(cudangine PUBLIC cxx_override)

#=======================================
#                SAMPLES
#=======================================

function(add_sample)
    set(options "")
    set(singleValues "")
    set(multipleValues SOURCES HEADERS)

    cmake_parse_arguments(SAMPLE "${options}" "${singleValues}" "${multipleValues}" ${ARGN})

    get_filename_component(cudangine_current_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    cuda_add_executable(${cudangine_current_name} ${SAMPLE_SOURCES} ${SAMPLE_HEADERS})
    set_target_properties(${cudangine_current_name} PROPERTIES CXX_STANDARD 14)
    target_link_libraries(${cudangine_current_name} cudangine)

    target_compile_features(${cudangine_current_name} PUBLIC cxx_noexcept)
    target_compile_features(${cudangine_current_name} PUBLIC cxx_override)
endfunction()

add_subdirectory("sample/basic")
add_subdirectory("sample/class")
add_subdirectory("sample/memory")
add_subdirectory("sample/stream")
