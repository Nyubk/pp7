import streamlit as st
import pandas as pd
import matplotlib
from matplotlib import matplotlib.pyplot as plt
import numpy as np
import sklearn

st.set_page_config(page_title="Simulador de Suplantación", layout="wide")

st.title("🔐 Simulador de Suplantación de Identidad en Redes Sociales")
st.markdown("Este simulador muestra cómo un ataque de suplantación puede propagarse según tus configuraciones.")

# Entradas del usuario
col1, col2 = st.columns(2)

with col1:
    contactos = st.slider("Número de contactos de la víctima", 10, 500, 100, 10)
    interaccion = st.slider("Nivel de interacción (1=bajo, 10=alto)", 1, 10, 5)
    seguridad = st.selectbox("Nivel de seguridad del perfil", ["Bajo", "Medio", "Alto"])
    
with col2:
    tiempo_respuesta = st.slider("Tiempo promedio de respuesta (en horas)", 1, 48, 6)
    dias = st.slider("Días simulados de propagación", 1, 14, 7)

# Parámetros para la simulación
seguridad_factor = {"Bajo": 1.5, "Medio": 1.0, "Alto": 0.6}[seguridad]
tasa_propagacion = (interaccion / 10) * seguridad_factor

# Simulación de propagación
propagacion = [1]  # Día 0: solo el atacante
for i in range(1, dias + 1):
    nuevos = propagacion[-1] * tasa_propagacion
    nuevos = min(nuevos, contactos - sum(propagacion))
    propagacion.append(propagacion[-1] + nuevos)

dias_simulados = list(range(dias + 1))
df = pd.DataFrame({"Día": dias_simulados, "Perfiles comprometidos": propagacion})

# Dashboard
st.subheader("📊 Resultados de la Simulación")

col3, col4, col5 = st.columns(3)
col3.metric("Perfiles comprometidos", f"{int(propagacion[-1])} / {contactos}")
col4.metric("Tasa de propagación", f"{tasa_propagacion:.2f} x día")
col5.metric("Prob. detección temprana", f"{100 - seguridad_factor * 30:.1f} %")

# Gráfico
fig, ax = plt.subplots()
ax.plot(df["Día"], df["Perfiles comprometidos"], marker='o', color='red')
ax.set_title("Propagación del ataque a lo largo del tiempo")
ax.set_xlabel("Día")
ax.set_ylabel("Perfiles comprometidos")
st.pyplot(fig)

# Recomendaciones automáticas
st.subheader("💡 Recomendaciones")
if tasa_propagacion > 1:
    st.error("⚠️ Alta tasa de propagación. Se recomienda reforzar medidas de seguridad.")
if seguridad == "Bajo":
    st.warning("🔐 Considera activar la verificación en dos pasos y restringir la visibilidad del perfil.")
else:
    st.success("✅ Buen nivel de seguridad, pero no bajes la guardia.")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===============================
# 🔎 GENERACIÓN DE DATOS SIMULADOS
# ===============================

# Simular dataset de perfiles
np.random.seed(42)
n_perfiles = 200
datos = pd.DataFrame({
    "contactos": np.random.randint(10, 500, n_perfiles),
    "interaccion": np.random.randint(1, 10, n_perfiles),
    "respuesta": np.random.randint(1, 48, n_perfiles),
    "seguridad_nivel": np.random.choice([0, 1, 2], n_perfiles),  # 0=bajo, 1=medio, 2=alto
})

# Etiquetar perfiles simulados con riesgo (0=no, 1=sí)
# Asumimos que riesgo aumenta si: baja seguridad, alta interacción, muchos contactos
datos["riesgo"] = (
    (datos["seguridad_nivel"] == 0).astype(int) +
    (datos["interaccion"] > 7).astype(int) +
    (datos["contactos"] > 300).astype(int)
)
datos["riesgo"] = (datos["riesgo"] >= 2).astype(int)

# ===============================
# 🤖 ENTRENAMIENTO DEL MODELO
# ===============================

X = datos[["contactos", "interaccion", "respuesta", "seguridad_nivel"]]
y = datos["riesgo"]

model = LogisticRegression()
model.fit(X, y)

# ===============================
# 💡 PREDICCIÓN PARA NUEVOS PERFILES
# ===============================

# Crear 10 perfiles ficticios con variaciones
perfiles_nuevos = pd.DataFrame({
    "contactos": np.random.randint(50, 500, 10),
    "interaccion": np.random.randint(1, 10, 10),
    "respuesta": np.random.randint(1, 48, 10),
    "seguridad_nivel": np.random.choice([0, 1, 2], 10)
})

# Predecir probabilidad de suplantación
perfiles_nuevos["prob_riesgo"] = model.predict_proba(perfiles_nuevos)[:, 1]
perfiles_nuevos["riesgo_estimado"] = perfiles_nuevos["prob_riesgo"].apply(lambda p: "Alto" if p > 0.7 else ("Medio" if p > 0.4 else "Bajo"))

# ===============================
# 📋 TABLA DE RESULTADOS
# ===============================
st.subheader("🔍 Perfiles Simulados y Riesgo de Suplantación")

st.dataframe(perfiles_nuevos.style
    .background_gradient(cmap="Reds", subset=["prob_riesgo"])
    .format({"prob_riesgo": "{:.2%}"})
)

# ===============================
# 🚨 ALERTAS AUTOMÁTICAS
# ===============================

st.subheader("🚨 Alertas Automáticas del Sistema")

riesgo_alto = perfiles_nuevos[perfiles_nuevos["riesgo_estimado"] == "Alto"]
riesgo_promedio = perfiles_nuevos["prob_riesgo"].mean()

if len(riesgo_alto) >= 5:
    st.error("🔴 Riesgo crítico: Más del 50% de los perfiles simulados presentan alto riesgo de suplantación. Recomendamos aplicar medidas urgentes.")
elif riesgo_promedio > 0.5:
    st.warning("🟠 Riesgo elevado en general. Monitorear perfiles y reforzar políticas de autenticación.")
elif len(riesgo_alto) == 0 and riesgo_promedio < 0.2:
    st.success("🟢 Riesgo bajo en los perfiles analizados. Buen nivel de seguridad.")

# Recomendaciones sugeridas
st.markdown("#### 🛡️ Recomendaciones del sistema:")
if riesgo_promedio > 0.5 or len(riesgo_alto) >= 5:
    st.markdown("- Activar verificación en dos pasos en redes sociales.")
    st.markdown("- Realizar auditorías periódicas de los perfiles.")
    st.markdown("- Implementar filtros automáticos de comportamiento sospechoso.")
    st.markdown("- Sensibilizar al personal sobre riesgos de ingeniería social.")
else:
    st.markdown("- Continuar monitoreando de forma preventiva.")
    st.markdown("- Reforzar las configuraciones de privacidad.")


# Pie de página
st.markdown("---")
st.caption("Proyecto de Ciencias de Datos para Negocios - 7° semestre | Simulación educativa")
