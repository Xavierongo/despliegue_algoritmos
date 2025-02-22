# Sistema de Clasificación de Vinos 🍷  

## ⚙️ Configuración Inicial  

### ruta local
cd C:\practica_d_a

2. Instalar dependencias
pip install -r requirements.txt


🚀 Ejecución del Proyecto
### Entrenamiento del Modelo
python src/main.py --n_estimators 150 --max_depth 5  

### Visualización de Resultados
mlflow ui --port 5000  

### Despliegue de la API
uvicorn src.api:app --reload  
