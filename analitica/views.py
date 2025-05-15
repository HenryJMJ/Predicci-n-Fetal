import matplotlib
matplotlib.use('Agg')  # Evita errores de GUI al usar matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os
import random
import networkx as nx
from django.db.models import Q
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render, get_object_or_404
from .forms import PrediccionIndividualForm
from .models import HistorialPrediccion

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from keras import Input

# Rutas
DATASET_PATH = os.path.join(settings.BASE_DIR, 'FGR_dataset.xlsx')
MODELOS_DIR = os.path.join(settings.BASE_DIR, 'analitica', 'modelos_entrenados')
os.makedirs(MODELOS_DIR, exist_ok=True)


def home(request):
    return render(request, 'analitica/home.html')


def entrenamiento(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo = request.FILES['archivo']
        df = pd.read_excel(archivo)
        df.dropna(inplace=True)
        df.columns = df.columns.str.strip()

        y = df["C31"]
        X = df.drop("C31", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Regresión Logística
        modelo_log = LogisticRegression(max_iter=2000, class_weight='balanced')
        modelo_log.fit(X_train, y_train)
        acc_log = accuracy_score(y_test, modelo_log.predict(X_test))
        joblib.dump(modelo_log, os.path.join(MODELOS_DIR, 'modelo_log.pkl'))

        # SVM
        modelo_svm = SVC(kernel='linear', probability=True)
        modelo_svm.fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, modelo_svm.predict(X_test))
        joblib.dump(modelo_svm, os.path.join(MODELOS_DIR, 'modelo_svm.pkl'))

        # Red Neuronal
        modelo_rna = Sequential()
        modelo_rna.add(Input(shape=(X_train.shape[1],)))
        modelo_rna.add(Dense(32, activation='relu'))
        modelo_rna.add(Dense(16, activation='relu'))
        modelo_rna.add(Dense(1, activation='sigmoid'))
        modelo_rna.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        modelo_rna.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        _, acc_rna = modelo_rna.evaluate(X_test, y_test, verbose=0)
        modelo_rna.save(os.path.join(MODELOS_DIR, 'modelo_rna.h5'))

        # Gráfica 1: Matriz de Confusión
        y_pred_log = modelo_log.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_log)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión - Regresión Logística")
        buffer_cm = io.BytesIO()
        plt.savefig(buffer_cm, format="png")
        buffer_cm.seek(0)
        image_cm = base64.b64encode(buffer_cm.getvalue()).decode("utf-8")
        plt.close()

        # Gráfica 2: Barras de exactitud
        modelos = ['Regresión Logística', 'SVM', 'Red Neuronal']
        exactitudes = [acc_log, acc_svm, acc_rna]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=modelos, y=exactitudes)
        plt.ylim(0, 1)
        plt.ylabel("Exactitud")
        plt.title("Comparación de Exactitudes")
        buffer_bar = io.BytesIO()
        plt.savefig(buffer_bar, format="png")
        buffer_bar.seek(0)
        image_bar = base64.b64encode(buffer_bar.getvalue()).decode("utf-8")
        plt.close()

        # Gráfica 3: Mapa Cognitivo Difuso
        G = nx.DiGraph()
        nodos = [f'C{i}' for i in range(1, 32)]
        G.add_nodes_from(nodos)

        for source in nodos:
            targets = random.sample(nodos, k=random.randint(1, 5))
            for target in targets:
                if source != target:
                    peso = round(random.uniform(-1, 1), 2)
                    G.add_edge(source, target, weight=peso)

        joblib.dump(G, os.path.join(MODELOS_DIR, 'mapa_cognitivo_difuso.pkl'))

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)
        edge_labels = {(u, v): f'{G[u][v]["weight"]}' for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Mapa Cognitivo Difuso - Relaciones entre variables")
        plt.axis('off')
        buffer_mcd = io.BytesIO()
        plt.savefig(buffer_mcd, format='png')
        buffer_mcd.seek(0)
        image_mcd = base64.b64encode(buffer_mcd.getvalue()).decode('utf-8')
        plt.close()

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

        return render(request, 'analitica/entrenamiento.html', contexto)

    return render(request, 'analitica/entrenamiento.html')


def prediccion_individual(request):
    resultado = None

    if request.method == 'POST':
        form = PrediccionIndividualForm(request.POST)

        if form.is_valid():
            import numpy as np
            import joblib
            import os
            import random
            import pandas as pd
            from keras.models import load_model

            df = pd.DataFrame([form.cleaned_data])

            mapeo = {
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

            df = df.rename(columns=mapeo)
            columnas_modelo = [f'C{i}' for i in range(1, 31)]
            for col in columnas_modelo:
                if col not in df.columns:
                    df[col] = 0
            df = df[columnas_modelo]

            MODELOS_DIR = os.path.join(os.path.dirname(__file__), 'modelos_entrenados')
            modelo_log = joblib.load(os.path.join(MODELOS_DIR, 'modelo_log.pkl'))
            modelo_svm = joblib.load(os.path.join(MODELOS_DIR, 'modelo_svm.pkl'))
            modelo_rna = load_model(os.path.join(MODELOS_DIR, 'modelo_rna.h5'))

            pred_log = modelo_log.predict(df)[0]
            pred_svm = modelo_svm.predict(df)[0]
            pred_rna = int(modelo_rna.predict(df)[0][0] > 0.5)

            resultado = {
                'Regresión Logística': 'FGR' if pred_log else 'Normal',
                'SVM': 'FGR' if pred_svm else 'Normal',
                'Red Neuronal': 'FGR' if pred_rna else 'Normal',
            }

            # --- Mapa Cognitivo Difuso ---
            pred_mcd = 'Modelo no entrenado aún'
            mcd_path = os.path.join(MODELOS_DIR, 'mapa_cognitivo_difuso.pkl')
            if os.path.exists(mcd_path):
                try:
                    G = joblib.load(mcd_path)
                    activacion = {f'C{i}': float(df.iloc[0][f'C{i}']) for i in range(1, 31)}

                    if 'C31' not in G.nodes:
                        G.add_node('C31')
                        for origen in random.sample(list(activacion.keys()), 5):
                            G.add_edge(origen, 'C31', weight=random.uniform(-1, 1))

                    for _ in range(2):
                        nueva_activacion = activacion.copy()
                        influencias = G.in_edges('C31', data=True)
                        suma = sum(activacion.get(origen, 0) * peso['weight'] for origen, _, peso in influencias)
                        nueva_activacion['C31'] = np.tanh(suma)
                        activacion.update(nueva_activacion)

                    valor_c31 = activacion.get('C31', 0)
                    pred_mcd = 'FGR' if valor_c31 > 0.1 else 'Normal'

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
        'prediccion': resultado
    })


def prediccion_lote(request):
    errores = []
    resultados = {}

    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo = request.FILES['archivo']
        ruta = default_storage.save('archivo_temporal.xlsx', archivo)

        try:
            df = pd.read_excel(ruta)
            if 'C31' not in df.columns:
                errores.append("El archivo debe contener la columna 'C31' como etiqueta.")
            else:
                X = df.drop('C31', axis=1)
                y = df['C31']

                # Cargar modelos supervisados
                modelo_log = joblib.load(os.path.join(MODELOS_DIR, 'modelo_log.pkl'))
                modelo_svm = joblib.load(os.path.join(MODELOS_DIR, 'modelo_svm.pkl'))
                modelo_rna = load_model(os.path.join(MODELOS_DIR, 'modelo_rna.h5'))

                columnas_esperadas = modelo_log.feature_names_in_
                if list(X.columns) != list(columnas_esperadas):
                    errores.append("Las columnas del archivo no coinciden con las esperadas por el modelo.")
                else:
                    # Predicciones supervisadas
                    pred_log = modelo_log.predict(X)
                    pred_svm = modelo_svm.predict(X)
                    pred_rna = (modelo_rna.predict(X) > 0.5).astype(int).flatten()

                    modelos = {
                        "Regresión Logística": pred_log,
                        "SVM": pred_svm,
                        "Red Neuronal": pred_rna
                    }

                    for nombre, pred in modelos.items():
                        acc = accuracy_score(y, pred)
                        cm = confusion_matrix(y, pred)

                        plt.figure(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'{nombre} - Matriz de Confusión')
                        plt.xlabel('Predicción')
                        plt.ylabel('Real')

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        plt.close()
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        resultados[nombre] = {
                            'exactitud': f"{acc:.2%}",
                            'imagen': img_base64
                        }

                    # === Mapa Cognitivo Difuso ===
                    mcd_path = os.path.join(MODELOS_DIR, 'mapa_cognitivo_difuso.pkl')
                    if os.path.exists(mcd_path):
                        G = joblib.load(mcd_path)

                        pred_mcd = []
                        for idx, fila in X.iterrows():
                            # Inicializar activación con valores de C1 a C30
                            activacion = {f'C{i}': float(fila[f'C{i}']) for i in range(1, 31)}

                            # Asegurar que C31 exista
                            if 'C31' not in G.nodes:
                                G.add_node('C31')
                                for origen in random.sample(list(activacion.keys()), 5):
                                    G.add_edge(origen, 'C31', weight=random.uniform(-1, 1))

                            # Propagación (2 pasos)
                            for _ in range(2):
                                nueva_activacion = activacion.copy()
                                influencias = G.in_edges('C31', data=True)
                                suma = sum(activacion.get(origen, 0) * peso['weight'] for origen, _, peso in influencias)
                                nueva_activacion['C31'] = np.tanh(suma)
                                activacion.update(nueva_activacion)

                            valor = activacion.get('C31', 0)
                            pred_mcd.append(1 if valor > 0.1 else 0)

                        acc = accuracy_score(y, pred_mcd)
                        cm = confusion_matrix(y, pred_mcd)

                        plt.figure(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
                        plt.title('Mapa Cognitivo Difuso - Matriz de Confusión')
                        plt.xlabel('Predicción')
                        plt.ylabel('Real')

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        plt.close()
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                        resultados["Mapa Cognitivo Difuso"] = {
                            'exactitud': f"{acc:.2%}",
                            'imagen': img_base64
                        }
                    else:
                        errores.append("El modelo de Mapa Cognitivo Difuso no está entrenado.")
        finally:
            default_storage.delete(ruta)

    return render(request, 'analitica/prediccion_lote.html', {
        'errores': errores,
        'resultados': resultados
    })


def historial_predicciones(request):
    historial = HistorialPrediccion.objects.all().order_by('-fecha')

    # Filtros
    edad_min = request.GET.get('edad_min')
    edad_max = request.GET.get('edad_max')
    fecha_min = request.GET.get('fecha_min')
    fecha_max = request.GET.get('fecha_max')

    if edad_min:
        historial = historial.filter(edad__gte=edad_min)
    if edad_max:
        historial = historial.filter(edad__lte=edad_max)
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