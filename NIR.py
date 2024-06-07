import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns

# Загрузка данных
def load_data():
    data = pd.read_csv('data/Student_Performance.csv')
    data = data.drop_duplicates()
    label_encoder = LabelEncoder()
    data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])
    return data

data = load_data()

st.title('Прогнозирование производительности студентов')

st.write('### Набор данных')
st.write(data.head())

# Масштабирование данных
scale_cols = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']
sc1 = MinMaxScaler()
sc1_data = sc1.fit_transform(data[scale_cols])

for i in range(len(scale_cols)):
    col = scale_cols[i]
    new_col_name = col + '_scaled'
    data[new_col_name] = sc1_data[:, i]

# Разделение данных на обучающую и тестовую выборки
data_dict = {"Previous Scores_scaled": data['Previous Scores_scaled'], "Hours Studied_scaled": data['Hours Studied_scaled']}
df_scaled = pd.DataFrame(data_dict)
X_train, X_test, y_train, y_test = train_test_split(df_scaled, data['Performance Index_scaled'], random_state=1)

# Выбор моделей
regr_models = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

model_name = st.selectbox('Выберите модель', list(regr_models.keys()))
model = regr_models[model_name]

st.write(f'### {model_name}')

# Обучение модели и предсказание
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'MAE: {mae:.3f}')
st.write(f'MSE: {mse:.3f}')
st.write(f'R2: {r2:.3f}')

# График зависимости производительности от количества изученных часов
hours_studied = st.sidebar.slider('Hours Studied', float(data['Hours Studied'].min()), float(data['Hours Studied'].max()), float(data['Hours Studied'].mean()))

avg_performance_by_hours = data.groupby('Hours Studied')['Performance Index'].mean()
fig, ax = plt.subplots()
ax.plot(avg_performance_by_hours.index, avg_performance_by_hours.values)
ax.axvline(hours_studied, color='r', linestyle='--')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Performance Index')
ax.set_title('Зависимость производительности от количества изученных часов')

st.pyplot(fig)

if st.checkbox('Показать корреляционную матрицу'):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data[scale_cols].corr(), annot=True, fmt='.2f', ax=ax)
    ax.set_title('Корреляционная матрица')
    st.pyplot(fig)


