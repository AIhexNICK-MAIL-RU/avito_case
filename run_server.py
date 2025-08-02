#!/usr/bin/env python3
"""
Скрипт для запуска веб-сервера с настройками для внешнего доступа
"""

import os
import socket
import subprocess
import sys
from app import app, load_model

def get_local_ip():
    """Получает локальный IP адрес компьютера"""
    try:
        # Создаем временное соединение для определения IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    print("🤖 Система анализа возвратов платежей")
    print("=" * 50)
    
    # Загружаем модель
    print("📦 Загрузка модели...")
    if load_model():
        print("✅ Модель успешно загружена")
    else:
        print("⚠️  Модель не найдена, будет создана при первом обучении")
    
    # Получаем IP адрес
    local_ip = get_local_ip()
    port = int(os.environ.get('PORT', 5000))
    
    print("\n🌐 Информация о доступе:")
    print(f"   Локальный доступ: http://localhost:{port}")
    print(f"   Внешний доступ:   http://{local_ip}:{port}")
    print(f"   Порт:            {port}")
    
    print("\n🔧 Настройки:")
    print("   - Для изменения порта: export PORT=8080")
    print("   - Для включения отладки: export FLASK_DEBUG=true")
    
    print("\n🚀 Запуск сервера...")
    print("   Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    # Запускаем приложение
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Отключаем отладку для продакшена
        threaded=True
    )

if __name__ == '__main__':
    main() 