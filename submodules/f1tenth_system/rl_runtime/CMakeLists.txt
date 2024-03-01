message(STATUS "=============== rl_runtime ==========================")
cmake_minimum_required(VERSION 3.20)
project(rl_runtime LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT TensorRT_ROOT)
    message(WARNING "TensorRT_ROOT not set, may fail to find TensorRT!")
endif()
cmake_policy(SET CMP0074 NEW)

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    f1tenth_msgs
    sensor_msgs
)

message("TensorRT Libraries: ${TensorRT_LIBRARIES}")

message(STATUS "CUDA version: ${CUDA_VERSION}")
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_72,code=sm_72
)

option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" ON)
if(USE_SYSTEM_SOPHUS)
    find_package(Sophus QUIET NO_MODULE)
endif()
if(NOT USE_SYSTEM_SOPHUS OR NOT TARGET Sophus::Sophus)
    set(USE_SYSTEM_SOPHUS OFF)
    include(${CMAKE_CURRENT_LIST_DIR}/sophus/sophus.cmake)
endif()

################################################################

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS
        f1tenth_msgs
        sensor_msgs
)

###########
## Build ##
###########
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(infer_v0_node
        src/cuda_common.cpp
        src/cuda_stream.cpp
        src/logger_trt.cpp
        src/trt_model.cpp
        src/infer_v0_node.cpp
        src/infer_v0.cpp
)
add_dependencies(infer_v0_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(infer_v0_node PRIVATE ${catkin_LIBRARIES} TensorRT::TensorRT CUDA::cudart fmt::fmt Sophus::Sophus)
set_target_properties(infer_v0_node PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(verify_trt_model
        src/cuda_common.cpp
        src/cuda_stream.cpp
        src/logger_trt.cpp
        src/trt_model.cpp
        src/verify_trt_model.cpp
        )
add_dependencies(verify_trt_model ${catkin_EXPORTED_TARGETS})
target_link_libraries(verify_trt_model PRIVATE ${catkin_LIBRARIES} TensorRT::TensorRT CUDA::cudart fmt::fmt Sophus::Sophus)
set_target_properties(verify_trt_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#############
## Install ##
#############
install(TARGETS infer_v0_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS verify_trt_model
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

message(STATUS "=============== rl_runtime ==========================")
