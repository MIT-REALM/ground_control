cmake_minimum_required(VERSION 3.0.0)
project(pose2odom)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    nav_msgs
    geometry_msgs
)

option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" ON)
if(USE_SYSTEM_SOPHUS)
    find_package(Sophus QUIET NO_MODULE)
endif()
if(NOT USE_SYSTEM_SOPHUS OR NOT TARGET Sophus::Sophus)
    set(USE_SYSTEM_SOPHUS OFF)
    include(${CMAKE_CURRENT_LIST_DIR}/sophus/sophus.cmake)
endif()

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS
    nav_msgs
    geometry_msgs
)

###########
## Build ##
###########
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(pose2odom_node
        src/pose2odom_node.cpp
        src/pose2odom.cpp
)
add_dependencies(pose2odom_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(pose2odom_node ${catkin_LIBRARIES} Sophus::Sophus)

#############
## Install ##
#############
install(TARGETS pose2odom_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
