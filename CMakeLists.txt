cmake_minimum_required(VERSION 3.16.3)
project(fpsCount)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(SYSTEM_ARCHITECTURE x64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(SYSTEM_ARCHITECTURE arm64)
endif()

link_directories(lib/${SYSTEM_ARCHITECTURE})
set(GIGE_CAM libMVSDK.so)



# add_executable(fpsCount framerate.cpp)
# target_link_libraries(fpsCount ${OpenCV_LIBRARIES})


# add_executable(newfpsCount newfps.cpp)
# target_link_libraries(newfpsCount ${OpenCV_LIBRARIES})

add_executable(gige_aruco src/gige_aruco.cpp)
target_link_libraries(gige_aruco ${GIGE_CAM} ${OpenCV_LIBRARIES})
