cmake_minimum_required(VERSION 3.1)

project(FastCorotDemo)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH})
set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")
set(CMAKE_MINSIZEREL_POSTFIX "_ms")

set(SOURCES
	main.cpp
	FastCorotFEM.cpp
	TetModel.cpp
	utilities/TetGenLoader.cpp
	utilities/MiniGL.cpp
)

set(HEADERS
	AVX_math.h
	FastCorotFEM.h
	Common.h
	utilities/TetGenLoader.h
	utilities/MiniGL.h
	utilities/Timing.h
	TetModel.h
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	add_compile_options(/arch:AVX)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	add_compile_options("-mavx")
endif()

set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/extern/eigen")
find_package( Eigen3 REQUIRED )

# Set include directories.
include_directories(
    ${EIGEN3_INCLUDE_DIR}
)

# OpenGL & GLEW library
find_package(OpenGL)

# executable
add_executable(FastCorotDemo
	${HEADERS}
    ${SOURCES}
	)
	
if (WIN32)
  subdirs(extern/freeglut)
  include_directories(${PROJECT_PATH}/extern/freeglut/include)
  
  set(GLUT_LIBRARIES freeglut)
  add_dependencies(FastCorotDemo freeglut)
else()
  find_package(GLUT REQUIRED)
endif()	
	
target_link_libraries(FastCorotDemo ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

add_definitions(-DDATA_PATH="${PROJECT_PATH}/meshes/")