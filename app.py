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
# ---------- Dictionnaire des descriptions des paramètres ----------
field_descriptions = {
    'Pregnancies': 'Nombre de grossesses',
    'Glucose': 'Taux de glucose (mg/dL)',
    'BloodPressure': 'Pression artérielle diastolique (mmHg)',
    'SkinThickness': 'Épaisseur du pli cutané tricipital (mm)',
    'Insulin': 'Insuline sérique (mu U/ml)',
    'BMI': 'Indice de masse corporelle (kg/m²)',
    'DiabetesPedigreeFunction': 'Fonction pedigree diabétique',
    'Age': 'Âge (années)',
    'Glucose_to_BMI': 'Ratio Glucose/IMC',
    'Insulin_to_Glucose': 'Ratio Insuline/Glucose',
    'Is_Elderly': 'Senior (>50 ans) : 0=Non, 1=Oui'
}

# ---------- Initialisation des états de session ----------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
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
# Ajout de features engineered
    data['Glucose_to_BMI'] = data['Glucose'] / data['BMI']
    data['Insulin_to_Glucose'] = data['Insulin'] / (data['Glucose'] + 1)
    data['Is_Elderly'] = (data['Age'] > 50).astype(int)
    
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
    "📊 Analyse du modèle",
    "🔬 Simulation médicale",
    "📋 Dossier patient",
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

# ---------- Page 1 : Prédiction ----------
if choice == "🩺 Prédiction":
    st.title("🩺 Prédiction du Risque Diabétique")
    st.markdown("### Saisissez vos paramètres médicaux pour évaluer votre risque")
    
    col1, col2 = st.columns(2)
    input_data = []
    
    # Configuration des champs avec descriptions
    field_descriptions = {
        'Pregnancies': 'Nombre de grossesses',
        'Glucose': 'Taux de glucose (mg/dL)',
        'BloodPressure': 'Pression artérielle diastolique (mmHg)',
        'SkinThickness': 'Épaisseur du pli cutané tricipital (mm)',
        'Insulin': 'Insuline sérique (mu U/ml)',
        'BMI': 'Indice de masse corporelle (kg/m²)',
        'DiabetesPedigreeFunction': 'Fonction pedigree diabétique',
        'Age': 'Âge (années)',
        'Glucose_to_BMI': 'Ratio Glucose/IMC',
        'Insulin_to_Glucose': 'Ratio Insuline/Glucose',
        'Is_Elderly': 'Senior (>50 ans) : 0=Non, 1=Oui'
    }
    
    with col1:
        for i, col in enumerate(columns[:len(columns)//2]):
            desc = field_descriptions.get(col, col)
            if col == 'Is_Elderly':
                val = st.selectbox(desc, [0, 1], key=f"input_{col}")
            else:
                val = st.number_input(desc, value=float(data[col].median()), min_value=0.0, step=0.1, key=f"input_{col}")
            input_data.append(val)
    
    with col2:
        for i, col in enumerate(columns[len(columns)//2:]):
            desc = field_descriptions.get(col, col)
            if col == 'Is_Elderly':
                val = st.selectbox(desc, [0, 1], key=f"input_{col}")
            else:
                val = st.number_input(desc, value=float(data[col].median()), min_value=0.0, step=0.1, key=f"input_{col}")
            input_data.append(val)

    if st.button("🔍 Analyser le risque", type="primary"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        if prediction == 1:
            st.error(f"⚠️ Risque élevé de diabète détecté (Probabilité: {probability[1]:.2%})")
            st.markdown("**Recommandations:**")
            st.markdown("- Consultez rapidement un médecin")
            st.markdown("- Adoptez une alimentation équilibrée")
            st.markdown("- Pratiquez une activité physique régulière")
        else:
            st.success(f"✅ Risque faible de diabète (Probabilité: {probability[0]:.2%})")
            st.markdown("**Conseils de prévention:**")
            st.markdown("- Maintenez un mode de vie sain")
            st.markdown("- Surveillez régulièrement votre glycémie")

# ---------- Page 2 : Analyse du modèle ----------
elif choice == "📊 Analyse du modèle":
    st.title("📊 Analyse du Modèle Prédictif")
    st.markdown("### Importance des variables dans la prédiction")
    
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Variable': columns, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    st.bar_chart(feature_df.set_index('Variable'))
    
    st.markdown("### Top 5 des facteurs les plus importants")
    for i, row in feature_df.head().iterrows():
        st.write(f"**{row['Variable']}**: {row['Importance']:.3f}")

# ---------- Page 3 : Simulation médicale ----------
elif choice == "🔬 Simulation médicale":
    st.title("🔬 Simulation d'Impact Clinique")
    st.markdown("### Analysez l'impact d'un paramètre spécifique")
    
    selected_feature = st.selectbox("Sélectionnez un paramètre clinique", columns)
    base_value = float(data[selected_feature].median())
    
    col1, col2 = st.columns(2)
    with col1:
        values = st.slider(f"Valeur de {selected_feature}", 
                          float(data[selected_feature].min()), 
                          float(data[selected_feature].max()), 
                          base_value, 0.1)
    
    with col2:
        st.metric("Valeur médiane normale", f"{base_value:.1f}")
        st.metric("Valeur sélectionnée", f"{values:.1f}")

    base_input = [data[col].median() for col in columns]
    base_input[list(columns).index(selected_feature)] = values
    input_array = np.array(base_input).reshape(1, -1)
    prediction = model.predict(scaler.transform(input_array))[0]
    probability = model.predict_proba(scaler.transform(input_array))[0]
    
    if prediction == 1:
        st.error(f"🔎 Résultat simulé : ⚠️ Risque élevé (Probabilité: {probability[1]:.2%})")
    else:
        st.success(f"🔎 Résultat simulé : ✅ Risque faible (Probabilité: {probability[0]:.2%})")

# ---------- Page 4 : Dossier patient ----------
elif choice == "📋 Dossier patient":
    st.title("📋 Dossier Médical Personnel")
    st.markdown("### Historique de vos analyses")
    
    if "patient_history" not in st.session_state:
        st.session_state.patient_history = []

    with st.expander("Nouvelle analyse", expanded=True):
        input_data = []
        cols = st.columns(2)
        
        for i, col in enumerate(columns):
            desc = field_descriptions.get(col, col)
            with cols[i % 2]:
                if col == 'Is_Elderly':
                    val = st.selectbox(desc, [0, 1], key=f"hist_{col}")
                else:
                    val = st.number_input(desc, value=float(data[col].median()), key=f"hist_{col}")
                input_data.append(val)

        if st.button("💾 Enregistrer l'analyse"):
            input_array = np.array(input_data).reshape(1, -1)
            pred = model.predict(scaler.transform(input_array))[0]
            prob = model.predict_proba(scaler.transform(input_array))[0]
            result = "Risque élevé" if pred == 1 else "Risque faible"
            
            analysis = {
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'data': input_data,
                'result': result,
                'probability': prob[1] if pred == 1 else prob[0]
            }
            
            st.session_state.patient_history.append(analysis)
            st.success("✅ Analyse enregistrée avec succès")

    # Afficher l'historique
    if st.session_state.patient_history:
        st.markdown("### Historique des analyses")
        history_df = pd.DataFrame([
            {
                'Date': analysis['date'],
                'Résultat': analysis['result'],
                'Probabilité': f"{analysis['probability']:.2%}"
            }
            for analysis in st.session_state.patient_history
        ])
        st.dataframe(history_df, use_container_width=True)



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



# ---------- Page 5 : À propos ----------
elif choice == "ℹ️ À propos":
    st.title("ℹ️ À Propos de DiabPredict")
    st.markdown("""
    ## 🏥 DiabPredict - Assistant Médical IA
    
    Cette application utilise l'intelligence artificielle pour :
    - **Prédire le risque de diabète** basé sur vos paramètres médicaux
    - **Analyser les facteurs de risque** les plus importants
    - **Simuler l'impact** de différents paramètres cliniques
    - **Fournir un chatbot médical** alimenté par Gemini
    
    ### 🤖 Chatbot IA
    Notre assistant virtuel utilise des modèles de langage avancés de Gemini pour répondre à vos questions sur le diabète.
    
    ### ⚠️ Avertissement Médical
    Cette application est uniquement à des fins éducatives et ne remplace pas un avis médical professionnel.
    
    ### 🔧 Technologies Utilisées
    - **Streamlit** pour l'interface utilisateur
    - **Scikit-learn** pour le machine learning
    - **Gemini API** pour l'IA conversationnelle
    - **Random Forest** pour la prédiction
    
    ---
    **Développé avec ❤️ pour la prévention du diabète**
       © 2025 DiabPredict - Tous droits réservés
    """)

# ---------- Pied de page ----------
st.markdown("""
<hr style="border:1px solid #2b5876; margin-top:50px;">
<div style="text-align:center; color:#666; font-size:0.9em;">
    Application médicale informative - Ne remplace pas une consultation
</div>
""", unsafe_allow_html=True)