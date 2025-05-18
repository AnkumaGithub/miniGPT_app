cmake_minimum_required(VERSION 3.20)
project(TextGenerator)

# Настройка vcpkg
set(CMAKE_TOOLCHAIN_FILE "C:/Users/79096/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_PREFIX_PATH "C:/Users/79096/vcpkg/installed/x64-mingw-dynamic")
set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic")

# Настройки стандарта C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Настройки для MinGW
if(MINGW)
    add_compile_options(
            -municode
            -mwindows
            -DUNICODE
            -D_UNICODE
    )
    add_link_options(
            -municode
            -mwindows
            -Wl,-subsystem=windows
    )
endif()

# Поиск зависимостей
find_package(cpr CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# Создание исполняемого файла (ОДИН РАЗ!)
add_executable(TextGenerator WIN32
        main.cpp
        src/imgui_impl_glfw.cpp
        src/imgui_impl_opengl3.cpp
        include/json.h
)

# Настройки include-директорий
target_include_directories(TextGenerator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Линковка библиотек
target_link_libraries(TextGenerator PRIVATE
        cpr::cpr
        CURL::libcurl
        glfw
        imgui::imgui
        OpenGL::GL
        user32
        gdi32
        shell32
        comdlg32
)
