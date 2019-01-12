cmake_minimum_required(VERSION 3.7.2)
project(pcb_bestukker)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} pcb_bestukker/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})



