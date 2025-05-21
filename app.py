
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Diabète Predictor", layout="wide")

# Chargement du modèle
@st.cache_data
def load_model():
    model = RandomForestClassifier()
    data = pd.read_csv("diabetes.csv")
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        data[col].replace(0, np.nan, inplace=True)
        data[col].fillna(data[col].median(), inplace=True)
    data['Glucose_to_BMI'] = data['Glucose'] / data['BMI']
    data['Insulin_to_Glucose'] = data['Insulin'] / data['Glucose']
    data['Is_Elderly'] = (data['Age'] > 50).astype(int)
    X = data.drop(['Outcome'], axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler, X.columns, data

model, scaler, columns, data = load_model()

# Logo
st.image("logo.png", width=120)

# Navigation horizontale
pages = ["🏠 Accueil", "📈 Modèle", "🔁 Scénario", "🗂️ Historique", "💬 ChatBot", "🎨 À propos"]
col_nav = st.columns(len(pages))
if "page" not in st.session_state:
    st.session_state.page = pages[0]

for i, p in enumerate(pages):
    if col_nav[i].button(p):
        st.session_state.page = p

# Contenus des pages
page = st.session_state.page

# Page 1 : Accueil (prédiction)
if page == "🏠 Accueil":
    st.title("🧠 Prédiction du Diabète")
    input_data = []
    for col in columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    if st.button("Prédire"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success("✅ Diabétique" if prediction == 1 else "❎ Non Diabétique")

# Page 2 : Explication du modèle
elif page == "📈 Modèle":
    st.title("📊 Explication du Modèle Random Forest")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': columns, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_df.set_index('Feature'))

# Page 3 : Simulation de scénario
elif page == "🔁 Scénario":
    st.title("🔍 Simulation : impact d'une variable")
    selected_feature = st.selectbox("Choisir une variable à modifier", columns)
    base_input = [data[col].median() for col in columns]
    value = st.slider(f"Valeur de {selected_feature}", 0.0, 200.0, float(base_input[columns.get_loc(selected_feature)]), 1.0)

    base_input[columns.get_loc(selected_feature)] = value
    input_array = np.array(base_input).reshape(1, -1)
    prediction = model.predict(scaler.transform(input_array))[0]
    st.write(f"🔎 Résultat simulé : {'✅ Diabétique' if prediction == 1 else '❎ Non Diabétique'}")

# Page 4 : Historique
elif page == "🗂️ Historique":
    st.title("🧾 Historique de prédictions")
    if "history" not in st.session_state:
        st.session_state.history = []

    input_data = []
    for col in columns:
        val = st.number_input(f"{col}", value=0.0, key=f"history_{col}")
        input_data.append(val)

    if st.button("Ajouter au journal"):
        input_array = np.array(input_data).reshape(1, -1)
        pred = model.predict(scaler.transform(input_array))[0]
        result = "Diabétique" if pred == 1 else "Non diabétique"
        st.session_state.history.append(input_data + [result])

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=list(columns) + ["Résultat"])
        st.dataframe(df)

# Page 5 : Chatbot
elif page == "💬 ChatBot":
    st.title("💬 ChatBot : Questions fréquentes")
    faq = {
        "Quels sont les symptômes du diabète ?":
            "Fatigue, soif excessive, mictions fréquentes, vision floue.",
        "Comment prévenir le diabète ?":
            "Avoir une alimentation équilibrée, faire de l'exercice régulièrement et surveiller son poids.",
        "Quel est un taux de glycémie normal ?":
            "À jeun : entre 70 et 99 mg/dL (3.9 à 5.5 mmol/L).",
        "Quels aliments éviter ?":
            "Évitez les sucres rapides, boissons sucrées, fast food, aliments ultra-transformés."
    }
    question = st.selectbox("Posez une question :", list(faq.keys()))
    if st.button("Répondre"):
        st.info(faq[question])

# Page 6 : À propos
elif page == "🎨 À propos":
    st.title("📝 À propos de ce projet")
    st.markdown("""
    Ce projet Streamlit prédit si un patient est diabétique à partir de données médicales.

    Réalisé avec 💡 par Lina Boussaada & Meriem Trabelsi
    Techniques utilisées : Random Forest, Normalisation, Feature Engineering, Clustering, Visualisation.

    Contact : [ing_idl@isi.com]
    """)
