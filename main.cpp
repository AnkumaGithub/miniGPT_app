#include <imgui.h>
#include <Windows.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <cpr/cpr.h>
#include <json.h>
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

// Простая реализация спиннера
void SimpleSpinner(float radius, float thickness, ImU32 color) {
    static float startTime = ImGui::GetTime();
    float time = ImGui::GetTime() - startTime;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImVec4 colorVec = ImColor(color).Value; // Используем ImColor.Value [[7]]

    const int segments = 12;
    const float angleStep = 2 * 3.141592 / segments;

    for (int i = 0; i < segments; i++) {
        float angle = i * angleStep + time * 2;
        float alpha = 1.0f - (float)i / segments;

        ImVec2 point(
            pos.x + radius + cos(angle) * radius,
            pos.y + radius + sin(angle) * radius
        );

        drawList->AddCircleFilled(
            point,
            thickness * alpha,
            IM_COL32(
                (int)(colorVec.x * 255),
                (int)(colorVec.y * 255),
                (int)(colorVec.z * 255),
                (int)(alpha * 255)
            )
        );
    }

    ImGui::Dummy(ImVec2(radius * 2, radius * 2));
}

void GenerateAsync(AppState& state) {
    state.isGenerating = true;
    state.requestFailed = false;

    std::thread([&state]() {
        try {
            std::string model = state.selectedModel == 0 ? "custom" : "gpt2";
            cpr::Response response;

            // Устанавливаем таймаут для запроса (5 секунд)
            response = cpr::Post(
                cpr::Url{"http://localhost:8000/generate"},
                cpr::Header{{"Content-Type", "application/json"}},
                cpr::Body{
                    R"({"prompt":")" + state.inputText +
                    R"(", "model_type":")" + model +
                    R"(", "max_tokens":200, "temperature":0.3})"
                },
                cpr::Timeout{30000} // Таймаут 30 секунд
            );

            // Проверяем статус ответа
            try {
                nlohmann::json json_response = nlohmann::json::parse(response.text);
                if (json_response.contains("generated_text")) {
                    state.outputText = json_response["generated_text"].get<std::string>();
                } else {
                    state.outputText = "Error: Invalid response format";
                }
            } catch (const std::exception& e) {
                state.outputText = "JSON Parse Error: " + std::string(e.what());
            }
        } catch (const std::exception& e) {
            state.outputText = "Exception: " + std::string(e.what());
            state.requestFailed = true;
        } catch (...) {
            state.outputText = "Unknown error";
            state.requestFailed = true;
        }
        state.isGenerating = false;
    }).detach();
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)  {
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    // Инициализация GLFW
    if (!glfwInit()) return 1;

    // Создание окна
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Text Generator", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    // Инициализация ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(15, 15); // Отступы внутри окна
    ImGui::StyleColorsDark();
    style.ItemSpacing = ImVec2(10, 15);    // Расстояние между элементами
    style.ScaleAllSizes(1.5f);

    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.29f, 0.48f, 0.54f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);

    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig config;
    config.OversampleH = 2;
    config.OversampleV = 2;
    io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/arial.ttf", 18.0f, &config, io.Fonts->GetGlyphRangesCyrillic());

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    AppState state;
    const char* models[] = { "Моя модель", "GPT-2" };

    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        char inputBuffer[1024] = {0}; // Буфер для ввода

        // Начало кадра ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Окно приложения
        ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver); // Начальный размер
        ImGui::Begin("Генератор текста", nullptr, ImGuiWindowFlags_None);

        // Выбор модели
        ImGui::Combo("Модель", &state.selectedModel, models, IM_ARRAYSIZE(models));
        ImGui::SameLine();

        if (state.isGenerating) {
            SimpleSpinner(15.0f, 6.0f, ImGui::GetColorU32(ImGuiCol_ButtonHovered));
        } else {
            ImGui::Dummy(ImVec2(15.0f * 2, 0.0f)); // Резервируем место под спиннер
        }

        // Поле ввода
        strncpy(inputBuffer, state.inputText.c_str(), IM_ARRAYSIZE(inputBuffer));
        ImGui::InputTextMultiline("##input", inputBuffer, IM_ARRAYSIZE(inputBuffer), ImVec2(-1, 150));
        state.inputText = inputBuffer;

        // Кнопка генерации
        if (ImGui::Button("Сгенерировать", ImVec2(-1, 40)) && !state.isGenerating) {
            GenerateAsync(state);
        }

        // Поле вывода
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_FrameBg]); // Наследуем фон поля ввода
        ImGui::PushStyleColor(ImGuiCol_Text, state.requestFailed ? IM_COL32(255, 0, 0, 255) : ImGui::GetColorU32(ImGuiCol_Text));

        ImGui::BeginChild("OutputScroll", ImVec2(-1, 300), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        ImGui::PushTextWrapPos(0.0f); // Перенос по ширине окна
        ImGui::TextUnformatted(state.outputText.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();

        ImGui::PopStyleColor(2);

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