#include <imgui.h>
#include <Windows.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <imguispinner.hpp>
#include <cpr/cpr.h>
#include <string>
#include <codecvt>
#include <thread>
#include <atomic>

std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

// Состояние интерфейса
struct AppState {
    std::string inputText;
    std::string outputText;
    int selectedModel = 0;
    std::atomic<bool> isGenerating{false};
    std::atomic<bool> requestFailed{false};
};


void GenerateAsync(AppState& state) {
    state.isGenerating = true;
    state.requestFailed = false;

    std::thread([&state]() {
        try {
            std::string model = state.selectedModel == 0 ? "custom" : "gpt2";

            // Ваш код для генерации
            auto response = cpr::Post(
                cpr::Url{"http://localhost:8000/generate"},
                cpr::Header{{"Content-Type", "application/json"}},
                cpr::Body{
                    R"({"prompt":")" + state.inputText +
                    R"(", "model_type":")" + model +
                    R"(", "max_tokens":200, "temperature":0.3})"
                }
            );

            if (response.status_code == 200) {
                state.outputText = response.text;
            } else {
                state.outputText = "Error: " + std::to_string(response.status_code);
                state.requestFailed = true;
            }
        } catch (...) {
            state.outputText = "Unknown error";
            state.requestFailed = true;
        }
        state.isGenerating = false;
    }).detach();
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)  {
    // Инициализация GLFW
    if (!glfwInit()) return 1;

    // Создание окна
    GLFWwindow* window = glfwCreateWindow(800, 600, "Text Generator", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    // Инициализация ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    AppState state;
    const char* models[] = { "Моя модель", "GPT-2" };

    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        char inputBuffer[1024] = {0}; // Буфер для ввода
        char outputBuffer[4096] = {0}; // Буфер для вывода

        // Начало кадра ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Окно приложения
        ImGui::Begin("Генератор текста", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

        // Выбор модели
        ImGui::Combo("Модель", &state.selectedModel, models, IM_ARRAYSIZE(models));

        // Поле ввода
        ImGui::InputTextMultiline("##input", inputBuffer, IM_ARRAYSIZE(inputBuffer), ImVec2(-1, 100));

        // Кнопка генерации
        if (ImGui::Button("Сгенерировать", ImVec2(-1, 0)) && !state.isGenerating) {
            GenerateAsync(state);
        }

        // Индикатор загрузки
        if (state.isGenerating) {
            ImGui::SameLine();
            ImSpinner::SpinnerRainbow("##spinner", 15.0f, 6.0f, 2.0f, ImGui::GetColorU32(ImGuiCol_ButtonHovered));
        }

        // Поле вывода
        ImGui::PushStyleColor(ImGuiCol_Text, state.requestFailed ? IM_COL32(255, 0, 0, 255) : ImGui::GetColorU32(ImGuiCol_Text));
        ImGui::InputTextMultiline("##output", outputBuffer, IM_ARRAYSIZE(outputBuffer), ImVec2(-1, -1), ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleColor();

        ImGui::End();

        // Рендеринг
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Очистка
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}