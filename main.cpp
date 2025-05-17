#include <windows.h>
#include <string>
#include <thread>

#define WM_UPDATE_UI (WM_USER + 1)

// Глобальные переменные для элементов интерфейса
HWND hInput, hOutput, hButton;

// Функция для запуска Python-скрипта
std::wstring RunPythonScript(const std::wstring& prompt) {
    std::wstring command = L"python gen.py --prompt \"" + prompt + L"\"";
    FILE* pipe = _wpopen(command.c_str(), L"r");
    if (!pipe) return L"Ошибка: не удалось запустить скрипт";

    wchar_t buffer[128];
    std::wstring result;
    while (fgetws(buffer, 128, pipe)) {
        result += buffer;
    }
    _pclose(pipe);
    return result;
}

// Асинхронная генерация текста
void GenerateAsync(HWND hWnd, const std::wstring& prompt) {
    std::wstring result = RunPythonScript(prompt);
    // Обновляем GUI из основного потока
    PostMessage(hWnd, WM_UPDATE_UI, 0, (LPARAM)new std::wstring(result));
}

// Оконная процедура
LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CREATE: {
            // Создание элементов интерфейса
            hInput = CreateWindowW(L"EDIT", L"", WS_BORDER | WS_VISIBLE | WS_CHILD,
                10, 10, 400, 30, hWnd, nullptr, nullptr, nullptr);

            hButton = CreateWindowW(L"BUTTON", L"Сгенерировать", WS_VISIBLE | WS_CHILD,
                10, 50, 120, 30, hWnd, (HMENU)1, nullptr, nullptr);

            hOutput = CreateWindowW(L"EDIT", L"", WS_BORDER | WS_VISIBLE | WS_CHILD | ES_MULTILINE | WS_VSCROLL,
                10, 90, 400, 200, hWnd, nullptr, nullptr, nullptr);
            return 0;
        }
        case WM_COMMAND: {
            if (LOWORD(wParam) == 1) { // Нажата кнопка
                wchar_t buffer[1024];
                GetWindowTextW(hInput, buffer, 1024);
                std::thread(GenerateAsync, hWnd, buffer).detach(); // Асинхронный вызов
            }
            return 0;
        }
        case WM_UPDATE_UI: { // Пользовательское сообщение для обновления интерфейса
            auto* result = reinterpret_cast<std::wstring*>(lParam);
            SetWindowTextW(hOutput, result->c_str());
            delete result;
            return 0;
        }
        case WM_DESTROY: {
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

// Точка входа
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
    // Регистрация класса окна
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"TextGeneratorClass";
    RegisterClassW(&wc);

    // Создание окна
    HWND hWnd = CreateWindowW(
        L"TextGeneratorClass", L"Генератор текста",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 440, 340,
        nullptr, nullptr, hInstance, nullptr
    );
    ShowWindow(hWnd, nCmdShow);

    // Цикл сообщений
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}