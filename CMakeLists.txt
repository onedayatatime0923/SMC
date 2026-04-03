cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Neural_Network_Verification)

############################################################
# Create compile flags
############################################################
set(CMAKE_CXX_FLAGS "-Wall -std=c++17 -pthread")
# set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17")
# SET(CMAKE_CXX_FLAGS "-static -Wall -O3 -std=c++17")
############################################################
# Create variables
############################################################
set(BASEPATH    ${PROJECT_SOURCE_DIR})
set(BINPATH     ${BASEPATH}/bin)
set(INCPATH     ${BASEPATH}/include)
set(LIBPATH     ${BASEPATH}/lib)
set(SRCPATH     ${BASEPATH}/src)
############################################################
# process variables
############################################################
IF (NOT EXISTS "${BINPATH}")
  FILE(MAKE_DIRECTORY ${BINPATH})
ENDIF()
IF (NOT EXISTS "${INCPATH}")
  FILE(MAKE_DIRECTORY ${INCPATH})
ENDIF()
IF (NOT EXISTS "${LIBPATH}")
  FILE(MAKE_DIRECTORY ${LIBPATH})
ENDIF()
IF (NOT EXISTS "${SRCPATH}")
  FILE(MAKE_DIRECTORY ${SRCPATH})
ENDIF()
############################################################
# find package               
############################################################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GUROBI REQUIRED)
find_package(OpenCV REQUIRED )
list(APPEND CMAKE_PREFIX_PATH "${BASEPATH}/tool/libtorch")
find_package(Torch REQUIRED)
find_package(Z3 REQUIRED)
find_package(ZLIB REQUIRED )
############################################################
# executable file output directory
############################################################
set(EXECUTABLE_OUTPUT_PATH ${BASEPATH})
############################################################
# include file
############################################################
# include directory
include_directories( ${INCPATH} )
include_directories( ${SRCPATH} )
include_directories( ${GUROBI_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${TORCH_INCLUDE_DIRS} )
include_directories( ${Z3_INCLUDE_DIRS} )
include_directories( ${ZLIB_INCLUDE_DIRS} )
############################################################
# library
############################################################
set(LIBRARY_OUTPUT_PATH ${LIBPATH})
# LIB
file(GLOB LIB ${LIBPATH}/*.a)
############################################################
# Create SRCLIB
############################################################
# SRC
file(GLOB SUBDIR RELATIVE ${SRCPATH} ${SRCPATH}/*)
foreach(dir ${SUBDIR})
    IF(IS_DIRECTORY ${SRCPATH}/${dir} AND NOT ${dir} MATCHES "main")
        file(GLOB FILE ${SRCPATH}/${dir}/*)
        set(SRC ${SRC} ${FILE})
    ENDIF()
endforeach()
add_library(srclib STATIC ${SRC})
target_link_libraries(srclib ${LIB} ${GUROBI_LIBRARIES} readline ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# ############################################################
# # Create an executable
# ############################################################
# MAINPATH
set(MAINPATH ${SRCPATH}/main/)

# Add an executable with the above sources
# add_executable(main ${MAINPATH}main.cpp)
add_executable(verification_property ${MAINPATH}nn_property.cpp)
# add_executable(verification_image ${MAINPATH}nn_image.cpp)
# add_executable(sdt_monolith ${MAINPATH}sdt_monolith.cpp)
add_executable(sdt_reachability ${MAINPATH}sdt_reachability.cpp)
# add_executable(nn_monolith ${MAINPATH}nn_monolith.cpp)
add_executable(nn_reachability ${MAINPATH}nn_reachability.cpp)
# add_executable(gurobi ${MAINPATH}gurobi.cpp)
# add_executable(test ${MAINPATH}test.cpp)

# link the new target with the target
# target_link_libraries(main srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
target_link_libraries(verification_property srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# target_link_libraries(verification_image srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# target_link_libraries(sdt_monolith srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
target_link_libraries(sdt_reachability srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# target_link_libraries(nn_monolith srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
target_link_libraries(nn_reachability srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# target_link_libraries(gurobi srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})
# target_link_libraries(test srclib ${LIB} ${GUROBI_LIBRARIES} readline ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Z3_LIBRARIES} ${ZLIB_LIBRARIES})

# link the new hello_library target with the hello_binary target
# set_target_properties(main PROPERTIES OUTPUT_NAME bin/NNV)
set_target_properties(verification_property PROPERTIES OUTPUT_NAME bin/VerificationProperty)
# set_target_properties(verification_image PROPERTIES OUTPUT_NAME bin/VerificationImage)
# set_target_properties(sdt_monolith PROPERTIES OUTPUT_NAME bin/SDTMonolith)
set_target_properties(sdt_reachability PROPERTIES OUTPUT_NAME bin/SDTReachability)
# set_target_properties(nn_monolith PROPERTIES OUTPUT_NAME bin/NNMonolith)
set_target_properties(nn_reachability PROPERTIES OUTPUT_NAME bin/NNReachability)
# set_target_properties(gurobi PROPERTIES OUTPUT_NAME bin/Gurobi)
# set_target_properties(test PROPERTIES OUTPUT_NAME bin/Test)
