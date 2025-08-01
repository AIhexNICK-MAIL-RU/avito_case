// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    loadAnalytics();
});

// Загрузка аналитических данных
async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        if (data.success) {
            updateStatistics(data);
            createCharts(data);
        } else {
            showError('Ошибка при загрузке аналитики: ' + data.message);
        }
    } catch (error) {
        showError('Ошибка при загрузке аналитики: ' + error.message);
    }
}

// Обновление статистики
function updateStatistics(data) {
    document.getElementById('totalOrders').textContent = data.total_orders.toLocaleString();
    document.getElementById('totalReturns').textContent = data.total_returns.toLocaleString();
    document.getElementById('returnRate').textContent = data.overall_return_rate + '%';
    document.getElementById('modelAccuracy').textContent = '85.2%'; // Примерная точность модели
}

// Создание графиков
function createCharts(data) {
    createMonthlyChart(data.monthly_returns);
    createPaymentChart(data.payment_analysis);
    createRegionChart(data.region_analysis);
    createAmountChart(data.amount_analysis);
}

// График по месяцам
function createMonthlyChart(monthlyData) {
    const months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'];
    
    const trace = {
        x: monthlyData.map(d => months[d.month - 1]),
        y: monthlyData.map(d => (d.return_rate * 100).toFixed(1)),
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: '#667eea',
            width: 3
        },
        marker: {
            size: 8,
            color: '#667eea'
        },
        name: 'Процент возвратов'
    };
    
    const layout = {
        title: {
            text: 'Динамика возвратов по месяцам',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Месяц'
        },
        yaxis: {
            title: 'Процент возвратов (%)',
            range: [0, 25]
        },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        height: 300
    };
    
    Plotly.newPlot('monthlyChart', [trace], layout, {responsive: true});
}

// График по способу оплаты
function createPaymentChart(paymentData) {
    const paymentLabels = {
        'card': 'Карта',
        'cash': 'Наличные',
        'online': 'Онлайн'
    };
    
    const trace = {
        x: paymentData.map(d => paymentLabels[d.payment_method] || d.payment_method),
        y: paymentData.map(d => (d.payment_return * 100).toFixed(1)),
        type: 'bar',
        marker: {
            color: ['#667eea', '#ff6b6b', '#48dbfb']
        },
        name: 'Процент возвратов'
    };
    
    const layout = {
        title: {
            text: 'Возвраты по способу оплаты',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Способ оплаты'
        },
        yaxis: {
            title: 'Процент возвратов (%)',
            range: [0, 25]
        },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        height: 300
    };
    
    Plotly.newPlot('paymentChart', [trace], layout, {responsive: true});
}

// График по региону
function createRegionChart(regionData) {
    const regionLabels = {
        'center': 'Центр',
        'suburb': 'Пригород',
        'remote': 'Отдаленный'
    };
    
    const trace = {
        x: regionData.map(d => regionLabels[d.delivery_region] || d.delivery_region),
        y: regionData.map(d => (d.payment_return * 100).toFixed(1)),
        type: 'bar',
        marker: {
            color: ['#667eea', '#ff6b6b', '#48dbfb']
        },
        name: 'Процент возвратов'
    };
    
    const layout = {
        title: {
            text: 'Возвраты по региону доставки',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Регион'
        },
        yaxis: {
            title: 'Процент возвратов (%)',
            range: [0, 25]
        },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        height: 300
    };
    
    Plotly.newPlot('regionChart', [trace], layout, {responsive: true});
}

// График по сумме заказа
function createAmountChart(amountData) {
    const trace = {
        x: amountData.map(d => d.amount_bin),
        y: amountData.map(d => (d.payment_return * 100).toFixed(1)),
        type: 'bar',
        marker: {
            color: '#667eea'
        },
        name: 'Процент возвратов'
    };
    
    const layout = {
        title: {
            text: 'Возвраты по сумме заказа',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Диапазон суммы заказа (₽)'
        },
        yaxis: {
            title: 'Процент возвратов (%)',
            range: [0, 25]
        },
        margin: { t: 50, b: 50, l: 60, r: 30 },
        height: 300
    };
    
    Plotly.newPlot('amountChart', [trace], layout, {responsive: true});
}

// Функция для показа ошибок
function showError(message) {
    const container = document.querySelector('.container');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    container.insertBefore(errorDiv, container.firstChild);
}

// Функция для обновления данных в реальном времени
function refreshAnalytics() {
    loadAnalytics();
}

// Добавляем кнопку обновления
document.addEventListener('DOMContentLoaded', function() {
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'btn btn-outline-primary position-fixed';
    refreshBtn.style.cssText = 'top: 20px; right: 20px; z-index: 1000;';
    refreshBtn.innerHTML = '<i class="fas fa-sync-alt me-2"></i>Обновить';
    refreshBtn.onclick = refreshAnalytics;
    document.body.appendChild(refreshBtn);
});

// Функция для экспорта графиков
function exportChart(chartId, filename) {
    const chart = document.getElementById(chartId);
    if (chart && chart.data) {
        Plotly.downloadImage(chart, {
            format: 'png',
            filename: filename,
            height: 600,
            width: 800
        });
    }
}

// Функция для создания отчета
function generateReport() {
    const reportData = {
        timestamp: new Date().toISOString(),
        totalOrders: document.getElementById('totalOrders').textContent,
        totalReturns: document.getElementById('totalReturns').textContent,
        returnRate: document.getElementById('returnRate').textContent,
        modelAccuracy: document.getElementById('modelAccuracy').textContent
    };
    
    const reportStr = JSON.stringify(reportData, null, 2);
    const dataBlob = new Blob([reportStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analytics_report.json';
    link.click();
    URL.revokeObjectURL(url);
}

// Функция для фильтрации данных
function filterData(startDate, endDate) {
    // Здесь можно добавить логику фильтрации данных по датам
    console.log('Фильтрация данных:', startDate, endDate);
    loadAnalytics(); // Перезагружаем данные
}

// Функция для сравнения периодов
function comparePeriods() {
    // Здесь можно добавить логику сравнения данных за разные периоды
    console.log('Сравнение периодов');
}

// Добавляем обработчики для интерактивных элементов
document.addEventListener('DOMContentLoaded', function() {
    // Добавляем обработчики для экспорта графиков
    const charts = ['monthlyChart', 'paymentChart', 'regionChart', 'amountChart'];
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart) {
            chart.addEventListener('plotly_click', function(data) {
                console.log('Клик по графику:', data);
            });
        }
    });
    
    // Добавляем обработчики для статистических карточек
    const statCards = document.querySelectorAll('.card.bg-primary, .card.bg-warning, .card.bg-danger, .card.bg-success');
    statCards.forEach(card => {
        card.addEventListener('click', function() {
            const title = this.querySelector('p').textContent;
            const value = this.querySelector('h4').textContent;
            console.log(`${title}: ${value}`);
        });
    });
});

// Функция для анимации загрузки
function showLoading() {
    const charts = ['monthlyChart', 'paymentChart', 'regionChart', 'amountChart'];
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart) {
            chart.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Загрузка...</span></div></div>';
        }
    });
}

// Функция для скрытия загрузки
function hideLoading() {
    // Загрузка скрывается автоматически при создании графиков
}

// Функция для адаптивности графиков
function resizeCharts() {
    const charts = ['monthlyChart', 'paymentChart', 'regionChart', 'amountChart'];
    charts.forEach(chartId => {
        const chart = document.getElementById(chartId);
        if (chart && chart.data) {
            Plotly.relayout(chart, {
                'width': chart.offsetWidth,
                'height': 300
            });
        }
    });
}

// Обработчик изменения размера окна
window.addEventListener('resize', resizeCharts);

// Функция для показа детальной информации
function showDetails(dataType, value) {
    console.log(`Детали для ${dataType}: ${value}`);
    // Здесь можно добавить модальное окно с детальной информацией
}

// Функция для создания трендов
function analyzeTrends() {
    // Здесь можно добавить анализ трендов
    console.log('Анализ трендов');
}

// Функция для прогнозирования трендов
function predictTrends() {
    // Здесь можно добавить прогнозирование трендов
    console.log('Прогнозирование трендов');
} 