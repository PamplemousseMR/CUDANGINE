#========================================
#	project http://www.icl.utk.edu/~mgates3/docs/cuda.html
#========================================
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT("CUDANGINE" CXX)

#================================================================================
#		init
#================================================================================

#========================================
#	variables
#========================================
SET(cudangine_lib_src_dir "src")
SET(cudangine_lib_header_dir "include")
SET(cudangine_lib_name "cudangine")

#========================================
#	compiler specifications
#========================================
SET(CMAKE_CXX_STANDARD 14)

#========================================
#	find files
#========================================
FILE(GLOB_RECURSE cudangine_lib_src_files ${cudangine_lib_src_dir}/*)
FILE(GLOB_RECURSE cudangine_lib_header_files ${cudangine_lib_header_dir}/*)

#================================================================================
#		create commun lib
#================================================================================

#========================================
#	find CUDA
#========================================
FIND_PACKAGE("CUDA")

IF(NOT ${CUDA_FOUND})
    MESSAGE(FATAL_ERROR "Can't find CUDA")
ENDIF()

#========================================
#	create lib
#========================================
CUDA_ADD_LIBRARY(${cudangine_lib_name}  ${cudangine_lib_src_files} ${cudangine_lib_header_files})
TARGET_INCLUDE_DIRECTORIES(${cudangine_lib_name} PUBLIC ${cudangine_lib_header_dir})
CUDA_INCLUDE_DIRECTORIES(${cudangine_lib_header_dir})

TARGET_COMPILE_FEATURES(${cudangine_lib_name} PUBLIC cxx_noexcept)
TARGET_COMPILE_FEATURES(${cudangine_lib_name} PUBLIC cxx_override)

#================================================================================
#		create
#================================================================================

#========================================
#	create one project by name
#========================================
FUNCTION(cudangineTest)

    #========================================
    #	variables
    #========================================
    SET(cudangine_src_dir "src")
    SET(cudangine_header_dir "include")

    #====================
    #	find files
    #====================
    FILE(GLOB_RECURSE cudangine_src_files ${cudangine_src_dir}/*)
    FILE(GLOB_RECURSE cudangine_header_files ${cudangine_header_dir}/*)

    #====================
    #	create project
    #====================
    GET_FILENAME_COMPONENT(cudangine_current_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

    CUDA_ADD_EXECUTABLE(${cudangine_current_name} ${cudangine_src_files} ${cudangine_header_files})
    TARGET_INCLUDE_DIRECTORIES(${cudangine_current_name} PRIVATE ${cudangine_header_dir} "../${cudangine_lib_header_dir}")
    CUDA_INCLUDE_DIRECTORIES(${cudangine_header_dir} "../${cudangine_lib_header_dir}")
    TARGET_LINK_LIBRARIES(${cudangine_current_name} PRIVATE ${cudangine_lib_name})

    TARGET_COMPILE_FEATURES(${cudangine_current_name} PUBLIC cxx_noexcept)
    TARGET_COMPILE_FEATURES(${cudangine_current_name} PUBLIC cxx_override)
ENDFUNCTION()

#========================================
#	find subdirectories
#========================================
SET(cudangine_project_dir "")

FILE(GLOB cudangine_glob_dir "./sample/*")
FOREACH(cudangine_dir ${cudangine_glob_dir})
    LIST(APPEND cudangine_project_dir ${cudangine_dir})
ENDFOREACH()

#========================================
#	create projects
#========================================
FOREACH(cudangine_dir ${cudangine_project_dir})
    MESSAGE(INFO " generate project : " ${cudangine_dir})
    ADD_SUBDIRECTORY(${cudangine_dir})
ENDFOREACH()
