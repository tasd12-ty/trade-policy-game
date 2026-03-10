cmake_minimum_required(VERSION 3.20)
project(eco_model_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTING "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Find required packages
find_package(Eigen3 3.4 REQUIRED)

# Optional: Ceres Solver for equilibrium solving
find_package(Ceres QUIET)
if(Ceres_FOUND)
    message(STATUS "Ceres Solver found - equilibrium solver enabled")
    add_compile_definitions(HAS_CERES)
else()
    message(STATUS "Ceres Solver not found - using Eigen LM fallback")
endif()

# Optional: nlohmann_json for I/O
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json found - JSON I/O enabled")
    add_compile_definitions(HAS_JSON)
endif()

# Library target
add_library(eco_model
    src/core/country_params.cpp
    src/core/model_params.cpp
    src/armington/ces.cpp
    src/armington/smooth_ops.cpp
    src/production/output.cpp
    src/production/cost.cpp
    src/production/income.cpp
    src/equilibrium/solver.cpp
    src/simulation/state.cpp
    src/simulation/country_sim.cpp
    src/simulation/two_country_sim.cpp
)

target_include_directories(eco_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(eco_model
    PUBLIC
        Eigen3::Eigen
)

if(Ceres_FOUND)
    target_link_libraries(eco_model PUBLIC Ceres::ceres)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(eco_model PUBLIC nlohmann_json::nlohmann_json)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(TARGETS eco_model
    EXPORT eco_model_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/eco_model
    DESTINATION include
)

install(EXPORT eco_model_targets
    FILE eco_model_targets.cmake
    NAMESPACE eco_model::
    DESTINATION lib/cmake/eco_model
)
