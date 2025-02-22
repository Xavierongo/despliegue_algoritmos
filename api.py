# api.py  
# Importación de bibliotecas para construcción de API y modelos de NLP  
from fastapi import FastAPI  
from pydantic import BaseModel  
from transformers import pipeline  
import numpy as np  

# Creación de instancia principal de FastAPI  
app = FastAPI()  

# ---------------------------------------------------------------  
# Simulación de modelo de clasificación (para propósitos de prueba)  
# ¡Reemplazar con carga de modelo real en producción!  
# ---------------------------------------------------------------  
class WineClassifier:  
    def predict(self, data: list):  
        # Genera predicción aleatoria temporal (0: clase 0, 1: clase 1, 2: clase 2)  
        return np.random.randint(0, 3)  

# -----------------------------------------------------------------  
# Pipelines preentrenados de Hugging Face para NLP  
# Modelos optimizados para equilibrio entre rendimiento y eficiencia  
# -----------------------------------------------------------------  
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text_generator = pipeline("text-generation", model="gpt2")

@app.get("/")
async def root():
    return {"message": "API para clasificación de vino"}

@app.post("/predict")
async def predict(features: list):
    classifier = WineClassifier()
    return {"class": int(classifier.predict(features))}

@app.post("/analyze_sentiment")
async def analyze_sentiment(text: str):
    return sentiment_analyzer(text)

@app.post("/generate_text")
async def generate_text(prompt: str):
    return text_generator(prompt, max_length=50)

@app.get("/model_info")
async def model_info():
    return {"model": "Random Forest", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}