# - Try to find ONNX
# Once done this will define
#  ONNX_FOUND - System has Gurobi
#  ONNX_INCLUDE_DIRS - The Gurobi include directories
#  ONNX_LIBRARIES - The libraries needed to use Gurobi

if (ONNX_INCLUDE_DIR)
    # in cache already
    set(ONNX_FOUND TRUE)
    set(ONNX_INCLUDE_DIRS "${ONNX_INCLUDE_DIR}" )
    set(ONNX_LIBRARIES "${ONNX_LIBRARY}" )
else (ONNX_INCLUDE_DIR)

    find_path(ONNX_INCLUDE_DIR 
        NAMES onnx.h
        PATHS "${BASEPATH}/tool/libonnx/src"
    )

    find_library( ONNX_LIBRARY 
        NAMES onnx
        PATHS "${BASEPATH}/tool/libonnx/src"
    )

    set(ONNX_INCLUDE_DIRS "${ONNX_INCLUDE_DIR}" )
    set(ONNX_LIBRARIES "${ONNX_LIBRARY}" )

    # use c++ headers as default
    # set(ONNX_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Gurobi Compiler Flags")

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(ONNX  DEFAULT_MSG ONNX_INCLUDE_DIR ONNX_LIBRARY )

    mark_as_advanced(ONNX_INCLUDE_DIR ONNX_LIBRARY)

endif(ONNX_INCLUDE_DIR)
