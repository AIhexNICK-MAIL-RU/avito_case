#!/bin/bash

# Скрипт для запуска веб-сервера
echo "🚀 Запуск веб-сервера для анализа возвратов платежей"

# Проверяем, активировано ли виртуальное окружение
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "📦 Активация виртуального окружения..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "❌ Виртуальное окружение не найдено. Создайте его командой:"
        echo "   python3 -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Проверяем зависимости
echo "🔍 Проверка зависимостей..."
pip install -r requirements.txt

# Устанавливаем порт (по умолчанию 5000)
export PORT=${PORT:-5000}

echo "🌐 Сервер будет доступен на порту $PORT"
echo "📱 Для доступа с других устройств используйте IP вашего компьютера"
echo ""

# Запускаем сервер
python run_server.py 