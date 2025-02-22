# funciones.py  
import argparse  

def parse_args():  
    """  
    Configura y parsea argumentos de línea de comandos para el entrenamiento del modelo.  
    Retorna:  
        Namespace: Objeto con parámetros configurados  
    """  
    parser = argparse.ArgumentParser(  
        description="Configura hiperparámetros del modelo Random Forest"  
    )  

    # -------------------------------------------------------------------------  
    # Hiperparámetros principales del modelo  
    # -------------------------------------------------------------------------  
    # Número de árboles en el bosque (valor típico: 100-200)  
    parser.add_argument(  
        "--n_estimators",   
        type=int,   
        default=100,  
        help="Cantidad de árboles en el bosque aleatorio"  
    )  

    # Profundidad máxima de cada árbol (None = ilimitado)  
    parser.add_argument(  
        "--max_depth",   
        type=int,   
        default=None,  
        help="Profundidad máxima por árbol. None para crecimiento libre"  
    )  

    # -------------------------------------------------------------------------  
    # Parámetro adicional para control de sobreajuste  
    # -------------------------------------------------------------------------  
    # Mínimas muestras requeridas para dividir un nodo  
    parser.add_argument(  
        "--min_samples_split",   
        type=int,   
        default=2,  
        help="Mínimo de muestras necesarias para dividir un nodo interno"  
    )  

    return parser.parse_args()  


### ###################################################################################3

def parse_args():
    parser = argparse.ArgumentParser(description="Configuración de entrenamiento")
    
    # Selección de modelo
    parser.add_argument("--model", type=str, default="rf",
                        choices=["rf", "svm"],
                        help="Algoritmo a usar: rf=Random Forest, svm=SVM")
    
    # Hiperparámetros comunes
    parser.add_argument("--random_state", type=int, default=42,
                        help="Semilla para reproducibilidad")
    
    # Parámetros para Random Forest
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Número de árboles (solo para RF)")
    
    # Parámetros para SVM
    parser.add_argument("--C", type=float, default=1.0,
                        help="Parámetro de regularización (solo para SVM)")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["linear", "poly", "rbf", "sigmoid"],
                        help="Tipo de kernel (solo para SVM)")

    return parser.parse_args()