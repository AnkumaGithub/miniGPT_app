//
// Created by 79096 on 25.05.2025.
//

#ifndef APP_H
#define APP_H

#pragma once

// Основные библиотеки
#include <imgui.h>
#include <Windows.h>
#include <../../include_headers/imgui_impl_glfw.h>
#include <../../include_headers/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <string>
#include <codecvt>
#include <thread>
#include <atomic>

// Глобальные переменные
extern std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

// Состояние приложения
struct AppState {
    std::string inputText;
    std::string outputText;
    int selectedModel = 0;
    std::atomic<bool> isGenerating{false};
    std::atomic<bool> requestFailed{false};

    int maxTokens = 200;
    float temperature = 0.3f;
};

// Функции
void SimpleSpinner(float radius, float thickness, ImU32 color);
void GenerateAsync(AppState& state);

#endif //APP_H
