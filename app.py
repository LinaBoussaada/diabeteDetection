import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# ---------- Initialisation du chatbot ----------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

def add_to_chat(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

# ---------- Données et Modèle ----------
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

# ---------- Configuration de la page ----------
st.set_page_config(
    page_title="DiabPredict - Prédiction de Diabète",
    page_icon="🏥",
    layout="wide"
)

# ---------- Logo et en-tête ----------
try:
    logo = Image.open("logo.png")
    st.sidebar.image(logo, width=200)
except:
    st.sidebar.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <h3 style="color:#2b5876;">DiabPredict</h3>
        <hr style="border:1px solid #2b5876;">
    </div>
    """, unsafe_allow_html=True)

# ---------- Navigation ----------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Menu", [
    "🩺 Prédiction",
    "📊 Analyse du modèle",
    "🔬 Simulation médicale",
    "📋 Dossier patient",
    "ℹ️ À propos"
])

# ---------- Bouton ChatBot flottant ----------
chat_button = st.sidebar.button("💬 Assistance Médicale", help="Poser une question à notre assistant virtuel")

if chat_button:
    st.session_state.show_chat = not st.session_state.show_chat

# ---------- Fonctionnalité du ChatBot ----------
def medical_chatbot():
    st.markdown("""
    <style>
    .chatbox {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 1000;
        padding: 15px;
        max-height: 500px;
        overflow-y: auto;
    }
    .chat-header {
        background: #2b5876;
        color: white;
        padding: 10px;
        border-radius: 8px 8px 0 0;
        margin: -15px -15px 10px -15px;
    }
    .user-message {
        background: #e3f2fd;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-message {
        background: #f1f1f1;
        padding: 8px 12px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.show_chat:
        st.markdown(f"""
        <div class="chatbox">
            <div class="chat-header">
                <h4>💬 Assistant Diabète</h4>
            </div>
        """, unsafe_allow_html=True)

        # Afficher l'historique du chat
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{chat["message"]}</div>', unsafe_allow_html=True)

        # Nouveau message
        user_input = st.text_input("Posez votre question...", key="chat_input", label_visibility="collapsed")
        
        if user_input:
            add_to_chat("user", user_input)
            
            # Réponses prédéfinies du chatbot
            responses = {
                "symptômes": "Les symptômes courants du diabète incluent soif excessive, mictions fréquentes, fatigue, vision floue et cicatrisation lente.",
                "prévention": "Pour prévenir le diabète : 1) Maintenez un poids santé 2) Faites de l'exercice régulièrement 3) Adoptez une alimentation équilibrée 4) Évitez le tabac",
                "alimentation": "Privilégiez les légumes verts, céréales complètes, protéines maigres. Évitez les sucres rapides et aliments transformés.",
                "urgence": "En cas de symptômes sévères (confusion, perte de conscience), contactez immédiatement les urgences médicales.",
                "diagnostic": "Le diagnostic se fait par test sanguin (glycémie à jeun, HbA1c). Consultez un médecin pour une évaluation précise."
            }
            
            # Trouver la meilleure réponse
            response = "Je suis un assistant médical virtuel. Pour des conseils personnalisés, veuillez consulter un professionnel de santé."
            for key in responses:
                if key in user_input.lower():
                    response = responses[key]
                    break
            
            add_to_chat("bot", response)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Page 1 : Prédiction ----------
if choice == "🩺 Prédiction":
    st.title("🩺 Prédiction du Risque Diabétique")
    
    col1, col2 = st.columns(2)
    input_data = []
    
    with col1:
        for col in columns[:len(columns)//2]:
            val = st.number_input(f"{col}", value=0.0, min_value=0.0, step=0.1)
            input_data.append(val)
    
    with col2:
        for col in columns[len(columns)//2:]:
            val = st.number_input(f"{col}", value=0.0, min_value=0.0, step=0.1)
            input_data.append(val)

    if st.button("🔍 Analyser le risque", type="primary"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            st.error("⚠️ Risque élevé de diabète détecté")
        else:
            st.success("✅ Risque faible de diabète")

# ---------- Page 2 : Analyse du modèle ----------
elif choice == "📊 Analyse du modèle":
    st.title("📊 Analyse du Modèle Prédictif")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Variable': columns, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_df.set_index('Variable'))

# ---------- Page 3 : Simulation médicale ----------
elif choice == "🔬 Simulation médicale":
    st.title("🔬 Simulation d'Impact Clinique")
    selected_feature = st.selectbox("Paramètre clinique", columns)
    base_value = float(data[selected_feature].median())
    values = st.slider(f"Valeur de {selected_feature}", 0.0, 200.0, base_value, 1.0)

    base_input = [data[col].median() for col in columns]
    base_input[columns.get_loc(selected_feature)] = values
    input_array = np.array(base_input).reshape(1, -1)
    prediction = model.predict(scaler.transform(input_array))[0]
    st.write(f"🔎 Résultat simulé : {'⚠️ Risque élevé' if prediction == 1 else '✅ Risque faible'}")

# ---------- Page 4 : Dossier patient ----------
elif choice == "📋 Dossier patient":
    st.title("📋 Dossier Médical Personnel")
    if "history" not in st.session_state:
        st.session_state.history = []

    input_data = []
    cols = st.columns(2)
    for i, col in enumerate(columns):
        with cols[i % 2]:
            val = st.number_input(f"{col}", value=0.0, key=f"hist_{col}")
            input_data.append(val)

    if st.button("💾 Enregistrer l'analyse"):
        input_array = np.array(input_data).reshape(1, -1)
        pred = model.predict(scaler.transform(input_array))[0]
        result = "Risque élevé" if pred == 1 else "Risque faible"
        st.session_state.history.append(input_data + [result])
        st.toast("Analyse enregistrée avec succès")

# ---------- Page 5 : À propos ----------
elif choice == "ℹ️ À propos":
    st.title("ℹ️ À Propos de DiabPredict")
    st.markdown("""
    Ce projet Streamlit prédit si un patient est diabétique à partir de données médicales.
    """)

# ---------- Afficher le chatbot sur toutes les pages ----------
medical_chatbot()
