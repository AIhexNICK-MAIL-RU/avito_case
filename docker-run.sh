#!/bin/bash

# Скрипт для запуска ML приложения в Docker контейнерах

echo "🐳 Запуск ML приложения в Docker контейнерах"
echo "=============================================="

# Проверяем, установлен ли Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker Desktop или Docker Engine."
    exit 1
fi

# Проверяем, установлен ли Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose."
    exit 1
fi

echo "✅ Docker и Docker Compose найдены"

# Останавливаем существующие контейнеры
echo "🛑 Остановка существующих контейнеров..."
docker-compose down

# Собираем образ
echo "🔨 Сборка Docker образа..."
docker-compose build --no-cache

# Запускаем контейнеры
echo "🚀 Запуск контейнеров..."
docker-compose up -d

# Ждем немного для запуска
echo "⏳ Ожидание запуска приложения..."
sleep 10

# Проверяем статус
echo "📊 Статус контейнеров:"
docker-compose ps

# Проверяем логи
echo "📋 Логи приложения:"
docker-compose logs ml-app

# Получаем IP адрес (работает на macOS и Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
else
    # Linux
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
fi

echo ""
echo "🌐 Приложение доступно по адресам:"
echo "   Локальный доступ: http://localhost:8080"
echo "   Внешний доступ:   http://${IP_ADDRESS}:8080"
echo ""
echo "🔧 Управление контейнерами:"
echo "   Остановить: docker-compose down"
echo "   Логи: docker-compose logs -f ml-app"
echo "   Перезапустить: docker-compose restart"
echo ""
echo "🎉 Приложение запущено в Docker контейнере!" 