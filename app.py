from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px

app = Flask(__name__)

# Глобальные переменные для модели
model = None
scaler = None
feature_names = None

def generate_sample_data(n_samples=1000):
    """Генерирует синтетические данные для демонстрации"""
    np.random.seed(42)
    
    # Параметры для генерации данных
    data = {
        'order_amount': np.random.normal(5000, 2000, n_samples),
        'delivery_distance': np.random.exponential(50, n_samples),
        'customer_age': np.random.normal(35, 10, n_samples),
        'payment_method': np.random.choice(['card', 'cash', 'online'], n_samples),
        'delivery_time': np.random.normal(3, 1, n_samples),
        'customer_rating': np.random.normal(4.2, 0.8, n_samples),
        'items_count': np.random.poisson(3, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples),
        'customer_orders_count': np.random.poisson(5, n_samples),
        'delivery_region': np.random.choice(['center', 'suburb', 'remote'], n_samples),
        'order_time_hour': np.random.randint(0, 24, n_samples),
        'weather_condition': np.random.choice(['sunny', 'rainy', 'cloudy'], n_samples),
        'traffic_level': np.random.choice(['low', 'medium', 'high'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Создаем целевую переменную (возврат платежа)
    # Логика: возврат более вероятен при определенных условиях
    return_prob = (
        (df['order_amount'] > 8000) * 0.3 +
        (df['delivery_distance'] > 100) * 0.2 +
        (df['customer_rating'] < 3.5) * 0.4 +
        (df['delivery_time'] > 5) * 0.3 +
        (df['payment_method'] == 'cash') * 0.1 +
        (df['is_weekend'] == 1) * 0.05 +
        (df['weather_condition'] == 'rainy') * 0.15 +
        (df['traffic_level'] == 'high') * 0.1
    )
    
    # Ограничиваем вероятность значениями [0, 1]
    return_prob = np.clip(return_prob, 0, 1)
    
    df['payment_return'] = np.random.binomial(1, return_prob)
    
    return df

def prepare_features(df):
    """Подготавливает признаки для модели"""
    # Создаем копию данных
    df_encoded = df.copy()
    
    # Убеждаемся, что все категориальные переменные присутствуют
    categorical_columns = ['payment_method', 'delivery_region', 'weather_condition', 'traffic_level']
    
    for col in categorical_columns:
        if col not in df_encoded.columns:
            # Если колонки нет, добавляем значение по умолчанию
            if col == 'payment_method':
                df_encoded[col] = 'card'
            elif col == 'delivery_region':
                df_encoded[col] = 'center'
            elif col == 'weather_condition':
                df_encoded[col] = 'sunny'
            elif col == 'traffic_level':
                df_encoded[col] = 'low'
    
    # Кодируем категориальные переменные
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns)
    
    # Убеждаемся, что все необходимые колонки присутствуют
    expected_columns = [
        'payment_method_card', 'payment_method_cash', 'payment_method_online',
        'delivery_region_center', 'delivery_region_suburb', 'delivery_region_remote',
        'weather_condition_sunny', 'weather_condition_rainy', 'weather_condition_cloudy',
        'traffic_level_low', 'traffic_level_medium', 'traffic_level_high'
    ]
    
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Выбираем числовые признаки
    numeric_features = [
        'order_amount', 'delivery_distance', 'customer_age', 
        'delivery_time', 'customer_rating', 'items_count',
        'customer_orders_count', 'order_time_hour'
    ]
    
    # Добавляем закодированные категориальные признаки
    categorical_features = [col for col in df_encoded.columns if col.startswith(('payment_method_', 'delivery_region_', 'weather_condition_', 'traffic_level_'))]
    
    feature_columns = numeric_features + categorical_features + ['is_weekend', 'is_holiday']
    
    # Убеждаемся, что все числовые признаки присутствуют
    for col in numeric_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Убеждаемся, что булевы признаки присутствуют
    if 'is_weekend' not in df_encoded.columns:
        df_encoded['is_weekend'] = 0
    if 'is_holiday' not in df_encoded.columns:
        df_encoded['is_holiday'] = 0
    
    return df_encoded[feature_columns], feature_columns

def train_model():
    """Обучает модель машинного обучения"""
    global model, scaler, feature_names
    
    # Генерируем данные
    df = generate_sample_data(2000)
    
    # Подготавливаем признаки
    X, feature_names = prepare_features(df)
    y = df['payment_return']
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Сохраняем модель
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    return model.score(X_test_scaled, y_test)

def load_model():
    """Загружает сохраненную модель"""
    global model, scaler, feature_names
    
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return True
    return False

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Обучает модель"""
    try:
        accuracy = train_model()
        return jsonify({
            'success': True,
            'accuracy': round(accuracy * 100, 2),
            'message': f'Модель успешно обучена! Точность: {accuracy*100:.2f}%'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при обучении модели: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Делает прогноз возврата платежа"""
    try:
        data = request.get_json()
        
        # Проверяем, загружена ли модель
        if model is None:
            if not load_model():
                return jsonify({
                    'success': False,
                    'message': 'Модель не обучена. Сначала обучите модель.'
                })
        
        # Создаем DataFrame с входными данными
        input_data = pd.DataFrame([data])
        
        # Подготавливаем признаки
        X, _ = prepare_features(input_data)
        
        # Проверяем, что все необходимые признаки присутствуют
        if len(X.columns) != len(feature_names):
            missing_features = set(feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(feature_names)
            return jsonify({
                'success': False,
                'message': f'Несоответствие признаков. Отсутствуют: {missing_features}, лишние: {extra_features}'
            })
        
        # Убеждаемся, что колонки в правильном порядке
        X = X[feature_names]
        
        # Масштабируем признаки
        X_scaled = scaler.transform(X)
        
        # Делаем прогноз
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]  # Вероятность возврата
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'message': f'Вероятность возврата платежа: {probability*100:.2f}%'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при прогнозировании: {str(e)}'
        })

@app.route('/analytics')
def analytics():
    """Страница с аналитикой"""
    return render_template('analytics.html')

@app.route('/api/analytics')
def get_analytics():
    """API для получения аналитических данных"""
    try:
        # Генерируем данные для аналитики
        df = generate_sample_data(1000)
        
        # Анализ по месяцам
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        df['month'] = df['date'].dt.month
        
        monthly_returns = df.groupby('month')['payment_return'].mean().reset_index()
        monthly_returns.columns = ['month', 'return_rate']
        
        # Анализ по сумме заказа
        df['amount_bin'] = pd.cut(df['order_amount'], bins=5)
        amount_analysis = df.groupby('amount_bin')['payment_return'].mean().reset_index()
        
        # Анализ по способу оплаты
        payment_analysis = df.groupby('payment_method')['payment_return'].mean().reset_index()
        
        # Анализ по региону доставки
        region_analysis = df.groupby('delivery_region')['payment_return'].mean().reset_index()
        
        return jsonify({
            'success': True,
            'monthly_returns': monthly_returns.to_dict('records'),
            'amount_analysis': amount_analysis.to_dict('records'),
            'payment_analysis': payment_analysis.to_dict('records'),
            'region_analysis': region_analysis.to_dict('records'),
            'total_orders': len(df),
            'total_returns': df['payment_return'].sum(),
            'overall_return_rate': round(df['payment_return'].mean() * 100, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при получении аналитики: {str(e)}'
        })

@app.route('/api/features')
def get_features():
    """API для получения списка признаков"""
    if feature_names is None:
        return jsonify({
            'success': False,
            'message': 'Модель не загружена'
        })
    
    return jsonify({
        'success': True,
        'features': feature_names
    })

if __name__ == '__main__':
    # Пытаемся загрузить модель при запуске
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001) 