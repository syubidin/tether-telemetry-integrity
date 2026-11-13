cmake_minimum_required(VERSION 3.16)
project(tether_core VERSION 1.0.0 LANGUAGES CXX)

add_library(tether_core SHARED src/core.cpp)
target_include_directories(tether_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tether_core PUBLIC cxx_std_17)
target_compile_definitions(tether_core PRIVATE TETHER_BUILDING_LIBRARY)
set_target_properties(tether_core PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

add_executable(tether_core_smoke src/smoke.cpp)
target_link_libraries(tether_core_smoke PRIVATE tether_core)
enable_testing()
add_test(NAME native_core_smoke COMMAND tether_core_smoke)
