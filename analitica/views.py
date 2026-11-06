import matplotlib
matplotlib.use('Agg')  # Evita errores de GUI al usar matplotlib

import base64
import io
import os

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import get_object_or_404, render
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .forms import PrediccionIndividualForm
from .models import HistorialPrediccion

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(settings.BASE_DIR, 'FGR_dataset.xlsx')
MODELOS_DIR = os.path.join(settings.BASE_DIR, 'analitica', 'modelos_entrenados')
os.makedirs(MODELOS_DIR, exist_ok=True)

SEMILLA = 42
UMBRAL_MCD = 0.1
COLUMNAS_MODELO = [f'C{i}' for i in range(1, 31)]

# Nota sobre el generador aleatorio:
# El Mapa Cognitivo Difuso (MCD) usa números aleatorios para inicializar pesos,
# NO con fines de seguridad. Por eso se usa un Generator de NumPy con semilla
# fija (reproducible) en lugar de `random` o `secrets`.


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _figura_a_base64(formato='png'):
    """Convierte la figura activa de matplotlib a base64 y la cierra."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format=formato, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _matriz_confusion_base64(y_real, y_pred, titulo, cmap='Blues', figsize=(4, 3), cbar=True):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=cbar)
    plt.title(titulo)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    return _figura_a_base64()


def construir_mapa_cognitivo(rng):
    """
    Construye el grafo del MCD con pesos aleatorios reproducibles.
    Garantiza que C31 reciba aristas entrantes (si no, la salida sería
    siempre tanh(0) = 0 y el modelo predeciría siempre 'Normal').
    """
    G = nx.DiGraph()
    nodos = [f'C{i}' for i in range(1, 32)]
    G.add_nodes_from(nodos)

    for source in nodos:
        k = int(rng.integers(1, 6))  # entre 1 y 5 destinos
        targets = rng.choice(nodos, size=k, replace=False)
        for target in targets:
            target = str(target)
            if source != target:
                peso = round(float(rng.uniform(-1, 1)), 2)
                G.add_edge(source, target, weight=peso)

    # Garantizar al menos 5 influencias entrantes hacia C31
    if G.in_degree('C31') < 5:
        origenes = rng.choice(COLUMNAS_MODELO, size=5, replace=False)
        for origen in origenes:
            origen = str(origen)
            if not G.has_edge(origen, 'C31'):
                G.add_edge(origen, 'C31', weight=round(float(rng.uniform(-1, 1)), 2))

    return G


def predecir_mcd(G, fila):
    """
    Devuelve la activación de C31 para una fila (Series o dict con C1..C30).
    C31 solo depende de sus aristas entrantes y de los valores de entrada,
    por lo que una sola pasada equivale a iterar varias veces.
    """
    suma = sum(
        float(fila[origen]) * datos['weight']
        for origen, _, datos in G.in_edges('C31', data=True)
        if origen in fila
    )
    return float(np.tanh(suma))


# ---------------------------------------------------------------------------
# Vistas básicas
# ---------------------------------------------------------------------------
def home(request):
    return render(request, 'analitica/home.html')


def entrenamiento(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo = request.FILES['archivo']
        df = pd.read_excel(archivo)
        df = df.dropna()
        df.columns = df.columns.str.strip()

        y = df["C31"]
        X = df.drop("C31", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEMILLA
        )

        # Regresión Logística
        modelo_log = LogisticRegression(max_iter=2000, class_weight='balanced')
        modelo_log.fit(X_train, y_train)
        y_pred_log = modelo_log.predict(X_test)
        acc_log = accuracy_score(y_test, y_pred_log)
        joblib.dump(modelo_log, os.path.join(MODELOS_DIR, 'modelo_log.pkl'))

        # SVM
        modelo_svm = SVC(kernel='linear', probability=True, random_state=SEMILLA)
        modelo_svm.fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, modelo_svm.predict(X_test))
        joblib.dump(modelo_svm, os.path.join(MODELOS_DIR, 'modelo_svm.pkl'))

        # Red Neuronal con scikit-learn
        modelo_rna = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu',
            max_iter=500, random_state=SEMILLA
        )
        modelo_rna.fit(X_train, y_train)
        acc_rna = accuracy_score(y_test, modelo_rna.predict(X_test))
        joblib.dump(modelo_rna, os.path.join(MODELOS_DIR, 'modelo_rna.pkl'))

        # Gráfica 1: Matriz de Confusión
        image_cm = _matriz_confusion_base64(
            y_test, y_pred_log,
            "Matriz de Confusión - Regresión Logística",
            cmap="Blues", figsize=(4, 4), cbar=False
        )

        # Gráfica 2: Barras de exactitud
        modelos = ['Regresión Logística', 'SVM', 'Red Neuronal']
        exactitudes = [acc_log, acc_svm, acc_rna]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=modelos, y=exactitudes)
        plt.ylim(0, 1)
        plt.ylabel("Exactitud")
        plt.title("Comparación de Exactitudes")
        image_bar = _figura_a_base64()

        # Gráfica 3: Mapa Cognitivo Difuso
        rng = np.random.default_rng(SEMILLA)
        G = construir_mapa_cognitivo(rng)
        joblib.dump(G, os.path.join(MODELOS_DIR, 'mapa_cognitivo_difuso.pkl'))

        pos = nx.spring_layout(G, seed=SEMILLA)
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)
        edge_labels = {(u, v): f'{G[u][v]["weight"]}' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Mapa Cognitivo Difuso - Relaciones entre variables")
        plt.axis('off')
        image_mcd = _figura_a_base64()

        contexto = {
            'resultado': f"""
                Regresión Logística: {acc_log:.2%}<br>
                SVM: {acc_svm:.2%}<br>
                Red Neuronal: {acc_rna:.2%}<br>
                Mapa Cognitivo Difuso: Visualización generada
            """,
            'grafica_cm': image_cm,
            'grafica_bar': image_bar,
            'grafica_mcd': image_mcd
        }

        # Invalidar la caché de modelos para que se carguen los recién entrenados
        _CACHE_MODELOS.clear()

        return render(request, 'analitica/entrenamiento.html', contexto)

    return render(request, 'analitica/entrenamiento.html')


# ---------------------------------------------------------------------------
# Carga de modelos (con caché en memoria)
# ---------------------------------------------------------------------------
_CACHE_MODELOS = {}


def _cargar_modelo(archivo, etiqueta):
    if archivo not in _CACHE_MODELOS:
        ruta = os.path.join(MODELOS_DIR, archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"{etiqueta} aún no ha sido entrenado. "
                "Ve a la página de entrenamiento primero."
            )
        # Seguridad: joblib.load usa pickle. Solo se cargan archivos
        # generados por esta misma aplicación.
        _CACHE_MODELOS[archivo] = joblib.load(ruta)
    return _CACHE_MODELOS[archivo]


def get_modelo_log():
    return _cargar_modelo('modelo_log.pkl', 'El modelo de Regresión Logística')


def get_modelo_svm():
    return _cargar_modelo('modelo_svm.pkl', 'El modelo SVM')


def get_modelo_rna():
    return _cargar_modelo('modelo_rna.pkl', 'El modelo Red Neuronal')


def get_mapa_cognitivo():
    return _cargar_modelo('mapa_cognitivo_difuso.pkl', 'El Mapa Cognitivo Difuso')


# ---------------------------------------------------------------------------
# Predicción individual
# ---------------------------------------------------------------------------
MAPEO_COLUMNAS = {
    'Age': 'C1', 'BMI': 'C2', 'Gestational_age_of_delivery': 'C3', 'Gravidity': 'C4', 'Parity': 'C5',
    'Initial_onset_symptoms': 'C6', 'Gestational_age_of_IOS_onset': 'C7', 'Interval_IOS_to_delivery': 'C8',
    'Gestational_age_hypertension_onset': 'C9', 'Interval_hypertension_to_delivery': 'C10',
    'Gestational_age_edema_onset': 'C11', 'Interval_edema_to_delivery': 'C12',
    'Gestational_age_proteinuria_onset': 'C13', 'Interval_proteinuria_to_delivery': 'C14',
    'Expectant_treatment': 'C15', 'Antihypertensive_before_hosp': 'C16', 'Past_history': 'C17',
    'Maximum_systolic_blood_pressure': 'C18', 'Maximum_diastolic_blood_pressure': 'C19',
    'Reasons_for_delivery': 'C20', 'Mode_of_delivery': 'C21', 'Maximum_BNP_value': 'C22',
    'Maximum_creatinine_value': 'C23', 'Maximum_uric_acid': 'C24', 'Maximum_proteinuria': 'C25',
    'Maximum_total_protein': 'C26', 'Maximum_albumin': 'C27', 'Maximum_ALT': 'C28',
    'Maximum_AST': 'C29', 'Maximum_platelet': 'C30',
}


def prediccion_individual(request):
    resultado = None
    error_modelo = None

    if request.method == 'POST':
        form = PrediccionIndividualForm(request.POST)

        if form.is_valid():
            df = pd.DataFrame([form.cleaned_data])
            df = df.rename(columns=MAPEO_COLUMNAS)

            for col in COLUMNAS_MODELO:
                if col not in df.columns:
                    df[col] = 0
            df = df[COLUMNAS_MODELO]

            try:
                modelo_log = get_modelo_log()
                modelo_svm = get_modelo_svm()
                modelo_rna = get_modelo_rna()
            except FileNotFoundError as e:
                error_modelo = str(e)
                return render(request, 'analitica/prediccion_individual.html', {
                    'form': form,
                    'prediccion': None,
                    'error_modelo': error_modelo,
                })

            pred_log = modelo_log.predict(df)[0]
            pred_svm = modelo_svm.predict(df)[0]
            pred_rna = modelo_rna.predict(df)[0]

            resultado = {
                'Regresión Logística': 'FGR' if pred_log else 'Normal',
                'SVM': 'FGR' if pred_svm else 'Normal',
                'Red Neuronal': 'FGR' if pred_rna else 'Normal',
            }

            # --- Mapa Cognitivo Difuso ---
            try:
                G = get_mapa_cognitivo()
                valor_c31 = predecir_mcd(G, df.iloc[0])
                pred_mcd = 'FGR' if valor_c31 > UMBRAL_MCD else 'Normal'
            except FileNotFoundError:
                pred_mcd = 'Modelo no entrenado aún'
            except Exception as e:
                pred_mcd = f'Error: {e}'

            resultado['Mapa Cognitivo Difuso'] = pred_mcd

            # Guardar en historial
            HistorialPrediccion.objects.create(
                nombre=form.cleaned_data['nombre'],
                C1=form.cleaned_data['Age'],
                C2=form.cleaned_data['BMI'],
                C3=form.cleaned_data['Gestational_age_of_delivery'],
                C4=form.cleaned_data['Gravidity'],
                C5=form.cleaned_data['Parity'],
                C6=form.cleaned_data['Initial_onset_symptoms'],
                C7=form.cleaned_data['Gestational_age_of_IOS_onset'],
                C8=form.cleaned_data['Interval_IOS_to_delivery'],
                C9=form.cleaned_data['Gestational_age_hypertension_onset'],
                C10=form.cleaned_data['Interval_hypertension_to_delivery'],
                C11=form.cleaned_data['Gestational_age_edema_onset'],
                C12=form.cleaned_data['Interval_edema_to_delivery'],
                C13=form.cleaned_data['Gestational_age_proteinuria_onset'],
                C14=form.cleaned_data['Interval_proteinuria_to_delivery'],
                C15=form.cleaned_data['Expectant_treatment'],
                C16=form.cleaned_data['Antihypertensive_before_hosp'],
                C17=form.cleaned_data['Past_history'],
                C18=form.cleaned_data['Maximum_systolic_blood_pressure'],
                C19=form.cleaned_data['Maximum_diastolic_blood_pressure'],
                C20=form.cleaned_data['Reasons_for_delivery'],
                C21=form.cleaned_data['Mode_of_delivery'],
                C22=form.cleaned_data['Maximum_BNP_value'],
                C23=form.cleaned_data['Maximum_creatinine_value'],
                C24=form.cleaned_data['Maximum_uric_acid'],
                C25=form.cleaned_data['Maximum_proteinuria'],
                C26=form.cleaned_data['Maximum_total_protein'],
                C27=form.cleaned_data['Maximum_albumin'],
                C28=form.cleaned_data['Maximum_ALT'],
                C29=form.cleaned_data['Maximum_AST'],
                C30=form.cleaned_data['Maximum_platelet'],
                resultado_log=resultado['Regresión Logística'],
                resultado_svm=resultado['SVM'],
                resultado_rna=resultado['Red Neuronal'],
                resultado_mcd=resultado['Mapa Cognitivo Difuso']
            )

    else:
        form = PrediccionIndividualForm()

    return render(request, 'analitica/prediccion_individual.html', {
        'form': form,
        'prediccion': resultado,
        'error_modelo': error_modelo,
    })


# ---------------------------------------------------------------------------
# Predicción por lote
# ---------------------------------------------------------------------------
def prediccion_lote(request):
    errores = []
    resultados = {}

    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo = request.FILES['archivo']
        ruta = default_storage.save('archivo_temporal.xlsx', archivo)

        try:
            df = pd.read_excel(ruta)
            df.columns = df.columns.str.strip()

            if 'C31' not in df.columns:
                errores.append("El archivo debe contener la columna 'C31' como etiqueta.")
            else:
                X = df.drop('C31', axis=1)
                y = df['C31']

                try:
                    modelo_log = get_modelo_log()
                    modelo_svm = get_modelo_svm()
                    modelo_rna = get_modelo_rna()
                except FileNotFoundError as e:
                    errores.append(str(e))
                    return render(request, 'analitica/prediccion_lote.html', {
                        'errores': errores,
                        'resultados': resultados
                    })

                columnas_esperadas = modelo_log.feature_names_in_
                if list(X.columns) != list(columnas_esperadas):
                    errores.append("Las columnas del archivo no coinciden con las esperadas por el modelo.")
                else:
                    # Predicciones supervisadas
                    modelos = {
                        "Regresión Logística": modelo_log.predict(X),
                        "SVM": modelo_svm.predict(X),
                        "Red Neuronal": modelo_rna.predict(X),
                    }

                    for nombre, pred in modelos.items():
                        resultados[nombre] = {
                            'exactitud': f"{accuracy_score(y, pred):.2%}",
                            'imagen': _matriz_confusion_base64(
                                y, pred, f'{nombre} - Matriz de Confusión', cmap='Blues'
                            ),
                        }

                    # === Mapa Cognitivo Difuso ===
                    try:
                        G = get_mapa_cognitivo()
                    except FileNotFoundError:
                        errores.append("El modelo de Mapa Cognitivo Difuso no está entrenado.")
                    else:
                        pred_mcd = [
                            1 if predecir_mcd(G, fila) > UMBRAL_MCD else 0
                            for _, fila in X.iterrows()
                        ]

                        resultados["Mapa Cognitivo Difuso"] = {
                            'exactitud': f"{accuracy_score(y, pred_mcd):.2%}",
                            'imagen': _matriz_confusion_base64(
                                y, pred_mcd,
                                'Mapa Cognitivo Difuso - Matriz de Confusión',
                                cmap='Oranges'
                            ),
                        }
        finally:
            default_storage.delete(ruta)

    return render(request, 'analitica/prediccion_lote.html', {
        'errores': errores,
        'resultados': resultados
    })


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def historial_predicciones(request):
    historial = HistorialPrediccion.objects.all().order_by('-fecha')

    # Filtros
    edad_min = request.GET.get('edad_min')
    edad_max = request.GET.get('edad_max')
    fecha_min = request.GET.get('fecha_min')
    fecha_max = request.GET.get('fecha_max')

    # La edad se guarda en el campo C1 del modelo
    if edad_min:
        historial = historial.filter(C1__gte=edad_min)
    if edad_max:
        historial = historial.filter(C1__lte=edad_max)
    if fecha_min:
        historial = historial.filter(fecha__date__gte=fecha_min)
    if fecha_max:
        historial = historial.filter(fecha__date__lte=fecha_max)

    return render(request, 'analitica/historial.html', {
        'historial': historial,
        'edad_min': edad_min,
        'edad_max': edad_max,
        'fecha_min': fecha_min,
        'fecha_max': fecha_max,
    })


def detalles_prediccion(request, id):
    prediccion = get_object_or_404(HistorialPrediccion, id=id)

    campos = {
        'Edad (C1)': prediccion.C1,
        'BMI (C2)': prediccion.C2,
        'Edad gestacional (C3)': prediccion.C3,
        'Gravidez (C4)': prediccion.C4,
        'Paridad (C5)': prediccion.C5,
        'Síntoma inicial (C6)': prediccion.C6,
        'Edad gestacional inicio síntoma (C7)': prediccion.C7,
        'Intervalo síntoma-parto (C8)': prediccion.C8,
        'Edad hipertensión (C9)': prediccion.C9,
        'Intervalo hipertensión-parto (C10)': prediccion.C10,
        'Edad edema (C11)': prediccion.C11,
        'Intervalo edema-parto (C12)': prediccion.C12,
        'Edad proteinuria (C13)': prediccion.C13,
        'Intervalo proteinuria-parto (C14)': prediccion.C14,
        'Tratamiento expectante (C15)': prediccion.C15,
        'Antihipertensivo previo (C16)': prediccion.C16,
        'Antecedente hipertensión (C17)': prediccion.C17,
        'Presión sistólica máx (C18)': prediccion.C18,
        'Presión diastólica máx (C19)': prediccion.C19,
        'Razón del parto (C20)': prediccion.C20,
        'Tipo de parto (C21)': prediccion.C21,
        'BNP máx (C22)': prediccion.C22,
        'Creatinina máx (C23)': prediccion.C23,
        'Ácido úrico máx (C24)': prediccion.C24,
        'Proteinuria máx (C25)': prediccion.C25,
        'Proteína total máx (C26)': prediccion.C26,
        'Albúmina máx (C27)': prediccion.C27,
        'ALT máx (C28)': prediccion.C28,
        'AST máx (C29)': prediccion.C29,
        'Plaquetas máx (C30)': prediccion.C30,
    }

    resultados = {
        'Regresión Logística': prediccion.resultado_log,
        'SVM': prediccion.resultado_svm,
        'Red Neuronal': prediccion.resultado_rna,
        'Mapa Cognitivo Difuso': prediccion.resultado_mcd,
    }

    return render(request, 'analitica/detalles_prediccion.html', {
        'prediccion': prediccion,
        'campos': campos,
        'resultados': resultados,
    })