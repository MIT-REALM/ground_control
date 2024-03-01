cmake_minimum_required(VERSION 3.0.0)
project(scan2pc)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    laser_geometry
)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS
    laser_geometry
    sensor_msgs
)

###########
## Build ##
###########
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(scan2pc_node
        src/scan2pc_node.cpp
        src/scan2pc.cpp
)
add_dependencies(scan2pc_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(scan2pc_node ${catkin_LIBRARIES})

#############
## Install ##
#############
install(TARGETS scan2pc_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
