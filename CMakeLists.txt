cmake_minimum_required(VERSION 3.20)
project(TextGenerator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MINGW)
    add_compile_options(-municode -mwindows)
    add_link_options(-municode -mwindows)
endif()

add_executable(TextGenerator main.cpp)

target_link_libraries(TextGenerator PRIVATE
        user32
        gdi32
)