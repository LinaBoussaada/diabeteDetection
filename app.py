import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import requests
from datetime import datetime
import google.generativeai as genai
genai.configure(api_key="AIzaSyCErMko7fBRNH4MMmdGvKimnBbgeISh7Bc")

modelgem = genai.GenerativeModel("models/gemini-1.5-flash")

# ---------- Initialisation des états de session ----------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

if 'hf_token' not in st.session_state:
    st.session_state.hf_token = None

# ---------- Fonctions utilitaires ----------
def add_to_chat(role, message):
    """Ajoute un message à l'historique de chat"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "role": role, 
        "message": message,
        "time": timestamp
    })

def get_medical_context():
    """Retourne le contexte médical pour le prompt"""
    contextmsg = """Vous êtes un assistant médical virtuel spécialisé en diabétologie. 
    Vous fournissez des informations précises et validées sur :
    - Les symptômes du diabète
    - La prévention et le dépistage
    - La gestion quotidienne de la maladie
    - Les complications potentielles
    
    Vous devez :
    1. Répondre de manière claire et compréhensible
    2. Toujours préciser que vos conseils ne remplacent pas un avis médical
    3. Recommander de consulter un professionnel de santé pour des cas personnels
    4. Vous baser sur les dernières recommandations médicales
    """
    modelgem.generate_content(contextmsg)



def querygem(message):
    get_medical_context()
    response = modelgem.generate_content(message)
    return response.text

# ---------- Modèle de prédiction de diabète ----------
@st.cache_data
def load_model():
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(70, 12, n_samples).clip(0, 120),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 60),
        'Insulin': np.random.normal(80, 115, n_samples).clip(0, 500),
        'BMI': np.random.normal(32, 7, n_samples).clip(18, 50),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples)
    })
    
    data['Outcome'] = (
        (data['Glucose'] > 140) | 
        (data['BMI'] > 35) | 
        (data['Age'] > 60)
    ).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = data.drop(['Outcome'], axis=1)
    y = data['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns, data

model, scaler, columns, data = load_model()

# ---------- Interface Streamlit ----------
st.set_page_config(
    page_title="DiabPredict - Prédiction de Diabète",
    page_icon="🏥",
    layout="wide"
)

# ---------- Barre latérale ----------
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <h2 style="color:#2b5876;">🏥 DiabPredict</h2>
    <p style="font-size:0.9em;">Assistant médical IA pour le diabète</p>
    <hr style="border:1px solid #2b5876;">
</div>
""", unsafe_allow_html=True)


# Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Menu", [
    "🩺 Prédiction",
    "📊 Analyse",
    "🤖 Assistant IA",
    "ℹ️ À propos"
])

# ---------- Styles CSS ----------
st.markdown("""
<style>
    /* Styles généraux */
    .stButton>button {
        border: 2px solid #2b5876;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    
    /* Styles spécifiques pour le chat */
    .chat-message {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    .message-time {
        font-size: 0.7em;
        color: #666;
        margin-top: 4px;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Page de prédiction ----------
if choice == "🩺 Prédiction":
    st.title("🩺 Évaluation du Risque Diabétique")
    st.markdown("""
    <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; margin-bottom:20px;">
        <b>Note :</b> Ce modèle prédictif est fourni à titre informatif uniquement. 
        Les résultats ne constituent pas un diagnostic médical.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Dictionnaire des descriptions des paramètres
    param_descriptions = {
        'Pregnancies': "Nombre de grossesses (0 si homme)",
        'Glucose': "Concentration de glucose plasmatique (mg/dL)",
        'BloodPressure': "Pression artérielle diastolique (mmHg)",
        'SkinThickness': "Épaisseur du pli cutané tricipital (mm)",
        'Insulin': "Insulinémie à jeun (μU/mL)",
        'BMI': "Indice de masse corporelle (kg/m²)",
        'DiabetesPedigreeFunction': "Fonction pedigree diabétique",
        'Age': "Âge (années)"
    }
    
    input_data = []
    with col1:
        for param in columns[:4]:
            input_data.append(st.number_input(
                param_descriptions.get(param, param),
                min_value=0.0,
                value=float(data[param].median()),
                key=f"input_{param}")
            )
    
    with col2:
        for param in columns[4:]:
            input_data.append(st.number_input(
                param_descriptions.get(param, param),
                min_value=0.0,
                value=float(data[param].median()),
                key=f"input_{param}"
            ))
    
    if st.button("Évaluer le risque", type="primary"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.markdown("---")
        st.subheader("Résultats de l'évaluation")
        
        if prediction == 1:
            st.error(f"⚠️ Risque élevé de diabète (Probabilité: {probability:.1%})")
            st.markdown("""
            **Recommandations :**
            - Consultez un médecin rapidement
            - Faites vérifier votre glycémie
            - Adoptez une alimentation équilibrée
            - Pratiquez une activité physique régulière
            """)
        else:
            st.success(f"✅ Risque faible de diabète (Probabilité: {1-probability:.1%})")
            st.markdown("""
            **Conseils de prévention :**
            - Maintenez un poids santé
            - Limitez les sucres ajoutés
            - Faites des bilans réguliers
            """)

# ---------- Page d'analyse ----------
elif choice == "📊 Analyse":
    st.title("📊 Analyse des Facteurs de Risque")
    
    # Importance des caractéristiques
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Paramètre': columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    st.subheader("Importance relative des paramètres")
    st.bar_chart(importance_df.set_index('Paramètre'))
    
    # Explications des paramètres
    st.subheader("Explications des paramètres")
    with st.expander("Détails des paramètres médicaux"):
        for param in columns:
            st.markdown(f"**{param}**")
            st.markdown(param_descriptions.get(param, "Pas de description disponible"))
            st.markdown(f"- Valeur médiane : {data[param].median():.1f}")
            st.markdown(f"- Plage typique : {data[param].min():.1f} à {data[param].max():.1f}")
            st.markdown("---")

# ---------- Page de l'assistant IA ----------
elif choice == "🤖 Assistant IA":
    st.title("🤖 Assistant Médical IA")
    st.markdown("""
    <div style="background-color:#fff8e1; padding:15px; border-radius:10px; margin-bottom:20px;">
        <b>Important :</b> Cet assistant IA fournit des informations générales sur le diabète. 
        Il ne remplace pas une consultation médicale.
    </div>
    """, unsafe_allow_html=True)
    
   
    # Affichage de l'historique du chat
    st.subheader("Conversation avec l'assistant")
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div><b>Vous</b></div>
                    <div>{msg["message"]}</div>
                    <div class="message-time">{msg["time"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div><b>Assistant</b></div>
                    <div>{msg["message"]}</div>
                    <div class="message-time">{msg["time"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone de saisie
    user_input = st.text_input(
        "Posez votre question sur le diabète...",
        placeholder="Ex: Quels sont les premiers symptômes du diabète ?",
        key="chat_input"
    )
    
    if st.button("Envoyer", type="primary") and user_input:
        add_to_chat("user", user_input)
        
        with st.spinner("L'assistant réfléchit..."):
            response = querygem(user_input)
            
            # Nettoyage de la réponse
            if isinstance(response, str):
                cleaned_response = response.strip()
                if not cleaned_response:
                    cleaned_response = "Je n'ai pas pu générer de réponse. Veuillez reformuler votre question."
            else:
                cleaned_response = str(response)
            
            # Ajout du disclaimer médical
            final_response = f"{cleaned_response}\n\n*Ceci est une information générale. Consultez un professionnel de santé pour un avis personnalisé.*"
            add_to_chat("bot", final_response)
            
            # Recharge la page pour afficher le nouveau message
            st.rerun()

# ---------- Page À propos ----------
elif choice == "ℹ️ À propos":
    st.title("ℹ️ À Propos de DiabPredict")
    
    st.markdown("""
    ## 🏥 DiabPredict - Assistant Médical IA
    
    **Version :** 2.0 (avec Gemini LLM)
    
    **Objectif :** Fournir des outils d'évaluation et d'information sur le diabète
    
    ### Fonctionnalités :
    - 🩺 Prédiction du risque diabétique
    - 📊 Analyse des facteurs de risque
    - 🤖 Assistant conversationnel spécialisé
    
    ### Technologies :
    - Modèle DeepSeek via Hugging Face
    - Random Forest pour la prédiction
    - Streamlit pour l'interface
    
    ### Avertissement :
    Cette application ne remplace pas un avis médical professionnel.
    Les résultats fournis sont à titre informatif uniquement.
    
    ### Développement :
    Application développée à des fins éducatives.
    
    © 2025 DiabPredict - Tous droits réservés
    """)

# ---------- Pied de page ----------
st.markdown("""
<hr style="border:1px solid #2b5876; margin-top:50px;">
<div style="text-align:center; color:#666; font-size:0.9em;">
    Application médicale informative - Ne remplace pas une consultation
</div>
""", unsafe_allow_html=True)