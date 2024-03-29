cmake_minimum_required(VERSION 2.8.3)
project(vesc_ackermann)

# Setting C++ standard to 11
if (NOT "${CMAKE_CXX_STANDARD_COMPUTED_DEFAULT}")
  message(STATUS "Changing CXX_STANDARD from C++98 to C++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif ("${CMAKE_CXX_STANDARD_COMPUTED_DEFAULT}" STREQUAL "98")
  message(STATUS "Changing CXX_STANDARD from C++98 to C++11")
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  nodelet
  pluginlib
  roscpp
  roslint
  std_msgs
  tf
  vesc_msgs
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
    geometry_msgs
    nav_msgs
    nodelet
    pluginlib
    std_msgs
    tf
    vesc_msgs
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# node executable
add_executable(vesc_to_odom_node src/vesc_to_odom_node.cpp
                                 src/vesc_to_odom.cpp)
add_dependencies(vesc_to_odom_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(vesc_to_odom_node
  ${catkin_LIBRARIES}
  Sophus::Sophus
)

add_executable(ackermann_to_vesc_node src/ackermann_to_vesc_node.cpp
                                      src/ackermann_to_vesc.cpp)
add_dependencies(ackermann_to_vesc_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(ackermann_to_vesc_node
  ${catkin_LIBRARIES}
)

# nodelet library
add_library(vesc_ackermann_nodelet src/ackermann_to_vesc_nodelet.cpp
                                   src/ackermann_to_vesc.cpp
                                   src/vesc_to_odom_nodelet.cpp
                                   src/vesc_to_odom.cpp)
add_dependencies(vesc_ackermann_nodelet ${catkin_EXPORTED_TARGETS})
target_link_libraries(vesc_ackermann_nodelet
  ${catkin_LIBRARIES}
  Sophus::Sophus
)

set(ROSLINT_CPP_OPTS "--filter=-build/c++11")
roslint_cpp()

#############
## Install ##
#############

install(TARGETS vesc_to_odom_node ackermann_to_vesc_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS vesc_ackermann_nodelet
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(FILES vesc_ackermann_nodelet.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  roslint_add_test()
endif()
