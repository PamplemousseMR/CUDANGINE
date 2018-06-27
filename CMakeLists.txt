CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# http://www.icl.utk.edu/~mgates3/docs/cuda.html
#------------ project ------------
PROJECT("CUDANGINE")

#------------ variables ------------
SET(cudangine_src_dir "src")
SET(cudangine_header_dir "include")
SET(cudangine_project_name "cudangine")

#------ find files ------
FILE(GLOB_RECURSE cudangine_src_files ${cudangine_src_dir}/*)
FILE(GLOB_RECURSE cudangine_header_files ${cudangine_header_dir}/*)

#------ find cuda ------
FIND_PACKAGE("CUDA")

IF(NOT ${CUDA_FOUND})
    MESSAGE(FATAL_ERROR "Can't find CUDA")
ENDIF()

#------------ create project ------------
INCLUDE_DIRECTORIES(${cudangine_header_dir})
CUDA_ADD_EXECUTABLE(${cudangine_project_name}  ${cudangine_src_files} ${cudangine_header_files})
