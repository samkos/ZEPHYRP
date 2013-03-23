PROJECT(test Fortran C)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

FIND_PACKAGE(PETSc)

include_directories(${PETSC_INCLUDES})

file(GLOB SRC . src/*.c)

include_directories(include)

ADD_EXECUTABLE(poissonC ${SRC})
TARGET_LINK_LIBRARIES(poissonC ${PETSC_LIBRARIES})
