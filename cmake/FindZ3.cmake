# - Try to find Z3
# Once done this will define
#  Z3_FOUND - System has Gurobi
#  Z3_INCLUDE_DIRS - The Gurobi include directories
#  Z3_LIBRARIES - The libraries needed to use Gurobi

if (Z3_INCLUDE_DIR)
  # in cache already
  set(Z3_FOUND TRUE)
  set(Z3_INCLUDE_DIRS "${Z3_INCLUDE_DIR}" )
  set(Z3_LIBRARIES "${Z3_LIBRARY}" )
else (Z3_INCLUDE_DIR)

find_path(Z3_INCLUDE_DIR 
          NAMES z3++.h
          PATHS "$ENV{Z3_HOME}/include"
          )

find_library( Z3_LIBRARY 
              NAMES z3
              PATHS "$ENV{Z3_HOME}/lib" 
              )

set(Z3_INCLUDE_DIRS "${Z3_INCLUDE_DIR}" )
set(Z3_LIBRARIES "${Z3_LIBRARY}" )

# use c++ headers as default
# set(Z3_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Gurobi Compiler Flags")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Z3  DEFAULT_MSG Z3_INCLUDE_DIR Z3_LIBRARY )

mark_as_advanced(Z3_INCLUDE_DIR Z3_LIBRARY)

endif(Z3_INCLUDE_DIR)
