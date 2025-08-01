// Глобальные переменные
let modelLoaded = false;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    checkModelStatus();
    setupEventListeners();
});

// Настройка обработчиков событий
function setupEventListeners() {
    // Обработчик формы прогнозирования
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });
}

// Проверка статуса модели
async function checkModelStatus() {
    try {
        const response = await fetch('/api/features');
        const data = await response.json();
        
        if (data.success) {
            modelLoaded = true;
            updateModelStatus('Модель загружена и готова к использованию', 'success');
        } else {
            updateModelStatus('Модель не обучена. Обучите модель для начала работы.', 'warning');
        }
    } catch (error) {
        updateModelStatus('Ошибка при проверке статуса модели', 'danger');
    }
}

// Обновление статуса модели
function updateModelStatus(message, type) {
    const statusElement = document.getElementById('modelStatus');
    const statusText = document.getElementById('modelStatusText');
    
    statusElement.className = `alert alert-${type}`;
    statusText.textContent = message;
}

// Обучение модели
async function trainModel() {
    const trainBtn = document.getElementById('trainBtn');
    const trainResult = document.getElementById('trainResult');
    
    // Показываем индикатор загрузки
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span class="loading"></span> Обучение...';
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            trainResult.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    ${data.message}
                </div>
            `;
            modelLoaded = true;
            updateModelStatus('Модель обучена и готова к использованию', 'success');
        } else {
            trainResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${data.message}
                </div>
            `;
        }
    } catch (error) {
        trainResult.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Ошибка при обучении модели: ${error.message}
            </div>
        `;
    } finally {
        // Восстанавливаем кнопку
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fas fa-play me-2"></i> Обучить модель';
    }
}

// Сбор данных формы
function collectFormData() {
    return {
        order_amount: parseFloat(document.getElementById('orderAmount').value),
        delivery_distance: parseFloat(document.getElementById('deliveryDistance').value),
        customer_age: parseInt(document.getElementById('customerAge').value),
        delivery_time: parseFloat(document.getElementById('deliveryTime').value),
        customer_rating: parseFloat(document.getElementById('customerRating').value),
        items_count: parseInt(document.getElementById('itemsCount').value),
        payment_method: document.getElementById('paymentMethod').value,
        delivery_region: document.getElementById('deliveryRegion').value,
        order_time_hour: parseInt(document.getElementById('orderTimeHour').value),
        customer_orders_count: parseInt(document.getElementById('customerOrdersCount').value),
        weather_condition: document.getElementById('weatherCondition').value,
        traffic_level: document.getElementById('trafficLevel').value,
        is_weekend: document.getElementById('isWeekend').checked ? 1 : 0,
        is_holiday: document.getElementById('isHoliday').checked ? 1 : 0
    };
}

// Валидация данных
function validateFormData(data) {
    const errors = [];
    
    if (data.order_amount <= 0) errors.push('Сумма заказа должна быть больше 0');
    if (data.delivery_distance < 0) errors.push('Расстояние доставки не может быть отрицательным');
    if (data.customer_age < 18 || data.customer_age > 100) errors.push('Возраст клиента должен быть от 18 до 100 лет');
    if (data.delivery_time < 0) errors.push('Время доставки не может быть отрицательным');
    if (data.customer_rating < 1 || data.customer_rating > 5) errors.push('Рейтинг клиента должен быть от 1 до 5');
    if (data.items_count < 1) errors.push('Количество товаров должно быть больше 0');
    if (data.order_time_hour < 0 || data.order_time_hour > 23) errors.push('Час заказа должен быть от 0 до 23');
    if (data.customer_orders_count < 0) errors.push('Количество заказов клиента не может быть отрицательным');
    
    return errors;
}

// Прогнозирование
async function makePrediction() {
    const predictBtn = document.getElementById('predictBtn');
    const predictionResult = document.getElementById('predictionResult');
    
    // Собираем данные формы
    const formData = collectFormData();
    
    // Валидируем данные
    const errors = validateFormData(formData);
    if (errors.length > 0) {
        predictionResult.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Ошибки валидации:</strong><br>
                ${errors.join('<br>')}
            </div>
        `;
        return;
    }
    
    // Показываем индикатор загрузки
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="loading"></span> Прогнозирование...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            const probability = data.probability;
            let resultClass = 'result-low';
            let icon = 'fa-check-circle';
            let message = 'Низкий риск возврата';
            
            if (probability > 50) {
                resultClass = 'result-high';
                icon = 'fa-exclamation-triangle';
                message = 'Высокий риск возврата';
            } else if (probability > 25) {
                resultClass = 'result-medium';
                icon = 'fa-exclamation-circle';
                message = 'Средний риск возврата';
            }
            
            predictionResult.innerHTML = `
                <div class="result-card ${resultClass}">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <h4><i class="fas ${icon} me-2"></i>${message}</h4>
                            <p class="mb-2"><strong>Вероятность возврата:</strong> ${probability}%</p>
                            <p class="mb-0"><strong>Прогноз:</strong> ${data.prediction ? 'Возврат ожидается' : 'Возврат не ожидается'}</p>
                        </div>
                        <div class="col-md-4 text-center">
                            <div class="display-4">${probability}%</div>
                            <small>риск возврата</small>
                        </div>
                    </div>
                </div>
            `;
        } else {
            predictionResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${data.message}
                </div>
            `;
        }
    } catch (error) {
        predictionResult.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Ошибка при прогнозировании: ${error.message}
            </div>
        `;
    } finally {
        // Восстанавливаем кнопку
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-magic me-2"></i> Сделать прогноз';
    }
}

// Функция для генерации случайных данных
function generateRandomData() {
    const randomData = {
        orderAmount: Math.floor(Math.random() * 15000) + 1000,
        deliveryDistance: Math.floor(Math.random() * 200) + 10,
        customerAge: Math.floor(Math.random() * 50) + 25,
        deliveryTime: (Math.random() * 8 + 1).toFixed(1),
        customerRating: (Math.random() * 3 + 2).toFixed(1),
        itemsCount: Math.floor(Math.random() * 10) + 1,
        orderTimeHour: Math.floor(Math.random() * 24),
        customerOrdersCount: Math.floor(Math.random() * 20) + 1
    };
    
    // Заполняем форму случайными данными
    document.getElementById('orderAmount').value = randomData.orderAmount;
    document.getElementById('deliveryDistance').value = randomData.deliveryDistance;
    document.getElementById('customerAge').value = randomData.customerAge;
    document.getElementById('deliveryTime').value = randomData.deliveryTime;
    document.getElementById('customerRating').value = randomData.customerRating;
    document.getElementById('itemsCount').value = randomData.itemsCount;
    document.getElementById('orderTimeHour').value = randomData.orderTimeHour;
    document.getElementById('customerOrdersCount').value = randomData.customerOrdersCount;
    
    // Случайные категориальные данные
    const paymentMethods = ['card', 'cash', 'online'];
    const regions = ['center', 'suburb', 'remote'];
    const weatherConditions = ['sunny', 'rainy', 'cloudy'];
    const trafficLevels = ['low', 'medium', 'high'];
    
    document.getElementById('paymentMethod').value = paymentMethods[Math.floor(Math.random() * paymentMethods.length)];
    document.getElementById('deliveryRegion').value = regions[Math.floor(Math.random() * regions.length)];
    document.getElementById('weatherCondition').value = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    document.getElementById('trafficLevel').value = trafficLevels[Math.floor(Math.random() * trafficLevels.length)];
    
    // Случайные булевы значения
    document.getElementById('isWeekend').checked = Math.random() > 0.7;
    document.getElementById('isHoliday').checked = Math.random() > 0.9;
}

// Добавляем кнопку для генерации случайных данных
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const randomBtn = document.createElement('button');
    randomBtn.type = 'button';
    randomBtn.className = 'btn btn-outline-secondary me-2';
    randomBtn.innerHTML = '<i class="fas fa-dice me-2"></i>Случайные данные';
    randomBtn.onclick = generateRandomData;
    
    const submitBtn = document.getElementById('predictBtn');
    submitBtn.parentNode.insertBefore(randomBtn, submitBtn);
});

// Функция для показа уведомлений
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Автоматически удаляем через 5 секунд
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Функция для экспорта данных
function exportData() {
    const formData = collectFormData();
    const dataStr = JSON.stringify(formData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'prediction_data.json';
    link.click();
    URL.revokeObjectURL(url);
}

// Функция для импорта данных
function importData(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const data = JSON.parse(e.target.result);
                // Заполняем форму импортированными данными
                Object.keys(data).forEach(key => {
                    const element = document.getElementById(key);
                    if (element) {
                        if (element.type === 'checkbox') {
                            element.checked = data[key] === 1;
                        } else {
                            element.value = data[key];
                        }
                    }
                });
                showNotification('Данные успешно импортированы', 'success');
            } catch (error) {
                showNotification('Ошибка при импорте данных', 'danger');
            }
        };
        reader.readAsText(file);
    }
} 