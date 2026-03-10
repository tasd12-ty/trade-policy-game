if(BUILD_EXAMPLES AND TARGET eco_model)
    add_executable(pipeline_demo pipeline_demo.cpp)
    target_link_libraries(pipeline_demo PRIVATE eco_model)
endif()
