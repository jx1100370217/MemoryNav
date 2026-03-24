cmake_minimum_required(VERSION 3.10)
project(fisheye_to_cylindrical)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(fisheye_to_cylindrical fisheye_to_cylindrical.cpp)
target_include_directories(fisheye_to_cylindrical PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fisheye_to_cylindrical
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
)
