# main.py
from funciones import parse_args
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow

def main():
    # Configuración de parámetros
    args = parse_args()
    
    # 1. Carga y preparación de datos
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=args.random_state,
        stratify=y
    )
    
    # 2. Preprocesamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Configuración de MLflow
    mlflow.set_experiment("Wine_Classification")
    
    with mlflow.start_run():
        # 4. Selección y entrenamiento del modelo
        if args.model == "rf":
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state
            )
            model_type = "Random Forest"
            
        elif args.model == "svm":
            model = SVC(
                C=args.C,
                kernel=args.kernel,
                random_state=args.random_state,
                probability=True
            )
            model_type = "SVM"
            
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluación
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 6. Registro de parámetros y métricas
        mlflow.log_params(vars(args))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("model_type", model_type)
        
        # 7. Visualizaciones (solo para Random Forest)
        if args.model == "rf":
            # Gráfico de importancia de características
            plt.figure(figsize=(10, 6))
            plt.barh(wine.feature_names, model.feature_importances_)
            plt.title("Importancia de Características - Random Forest")
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            
        # Matriz de confusión para ambos modelos
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=wine.target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - {model_type}")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # 8. Registro del modelo
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()