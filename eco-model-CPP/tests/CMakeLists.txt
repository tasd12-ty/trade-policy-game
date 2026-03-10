if(BUILD_TESTING AND TARGET eco_model)
    find_package(GTest QUIET)
    
    if(GTest_FOUND)
        # 使用系统安装的 GTest
        message(STATUS "Google Test found")
    else()
        # 使用 FetchContent 获取 GTest
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # 测试可执行文件
    add_executable(eco_model_tests
        test_armington.cpp
        test_production.cpp
    )
    
    target_link_libraries(eco_model_tests
        PRIVATE
            eco_model
            GTest::gtest_main
    )
    
    include(GoogleTest)
    gtest_discover_tests(eco_model_tests)
endif()
