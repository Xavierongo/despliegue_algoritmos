# Sistema de Clasificación de Vinos 🍷  

## ⚙️ Configuración Inicial  

### ruta local
cd C:\practica_d_a

### Instalar dependencias
pip install -r requirements.txt


## Ejecución del Proyecto
### Entrenamiento del Modelo
# Entrenar Random Forest
python main.py --model rf --n_estimators 150

# Entrenar SVM
python main.py --model svm --kernel poly

### Visualización de Resultados
mlflow ui --port 5000  

### Despliegue de la API
uvicorn src.api:app --reload  

### ejecutar SVM
python src/main.py --model svm --C 0.8 --kernel linear
