# 🐳 Docker развертывание ML приложения

## ✅ Быстрый запуск

### 1. Установка Docker

#### macOS:
```bash
# Установите Docker Desktop с официального сайта
# https://www.docker.com/products/docker-desktop
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

### 2. Запуск приложения

#### Вариант 1: Использование скрипта (рекомендуется)
```bash
./docker-run.sh
```

#### Вариант 2: Ручной запуск
```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f ml-app
```

## 🔧 Управление контейнерами

### Основные команды:

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Просмотр логов
docker-compose logs -f ml-app

# Статус контейнеров
docker-compose ps

# Пересборка образа
docker-compose build --no-cache
```

### Управление отдельными контейнерами:

```bash
# Остановить только ML приложение
docker-compose stop ml-app

# Запустить только ML приложение
docker-compose start ml-app

# Перезапустить только ML приложение
docker-compose restart ml-app
```

## 🌐 Доступ к приложению

После запуска приложение будет доступно по адресам:

- **Локальный доступ**: http://localhost:8080
- **Внешний доступ**: http://[ВАШ_IP]:8080

### Проверка работы:
```bash
# Проверка API
curl http://localhost:8080/api/features

# Проверка главной страницы
curl http://localhost:8080/
```

## 📁 Структура Docker файлов

```
├── Dockerfile              # Конфигурация Docker образа
├── docker-compose.yml      # Конфигурация сервисов
├── .dockerignore          # Исключения для Docker
├── docker-run.sh          # Скрипт запуска
└── DOCKER_README.md       # Эта инструкция
```

## ⚙️ Настройки

### Изменение порта:
Отредактируйте `docker-compose.yml`:
```yaml
ports:
  - "9000:8080"  # Внешний порт 9000, внутренний 8080
```

### Переменные окружения:
```yaml
environment:
  - FLASK_DEBUG=true
  - PORT=8080
```

### Монтирование файлов:
```yaml
volumes:
  - ./data:/app/data  # Монтирование папки с данными
```

## 🔍 Отладка

### Просмотр логов:
```bash
# Все логи
docker-compose logs

# Логи конкретного сервиса
docker-compose logs ml-app

# Логи в реальном времени
docker-compose logs -f ml-app
```

### Вход в контейнер:
```bash
# Войти в контейнер
docker-compose exec ml-app bash

# Выполнить команду в контейнере
docker-compose exec ml-app python -c "print('Hello from container')"
```

### Проверка состояния:
```bash
# Статус контейнеров
docker-compose ps

# Использование ресурсов
docker stats
```

## 🚀 Продакшн развертывание

### 1. Настройка Nginx (опционально):
Раскомментируйте секцию nginx в `docker-compose.yml`

### 2. Настройка SSL (опционально):
Добавьте SSL сертификаты и настройте HTTPS

### 3. Мониторинг:
```bash
# Установите Prometheus и Grafana для мониторинга
```

## 🛠️ Устранение проблем

### Проблема: Порт занят
```bash
# Найдите процесс, использующий порт
lsof -i :8080

# Остановите процесс или измените порт в docker-compose.yml
```

### Проблема: Контейнер не запускается
```bash
# Проверьте логи
docker-compose logs ml-app

# Проверьте статус
docker-compose ps
```

### Проблема: Модели не загружаются
```bash
# Проверьте, что файлы моделей существуют
ls -la *.pkl

# Пересоберите образ
docker-compose build --no-cache
```

## 📊 Мониторинг и логи

### Health Check:
Контейнер автоматически проверяет здоровье приложения каждые 30 секунд.

### Логи:
- Логи приложения: `docker-compose logs ml-app`
- Логи системы: `docker system logs`

### Метрики:
```bash
# Использование ресурсов
docker stats

# Размер образов
docker images
```

## 🔒 Безопасность

### Рекомендации:
1. Используйте не-root пользователя (уже настроено)
2. Обновляйте базовые образы регулярно
3. Сканируйте образы на уязвимости
4. Используйте secrets для конфиденциальных данных

### Сканирование уязвимостей:
```bash
# Установите Trivy
brew install trivy

# Сканируйте образ
trivy image avito-ml-app
```

---

**🎉 Ваше ML приложение теперь работает в Docker контейнерах!** 