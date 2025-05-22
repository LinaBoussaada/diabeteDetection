import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
import json
import time

# ---------- Configuration du chatbot Hugging Face ----------
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
# Vous pouvez aussi utiliser d'autres modèles gratuits comme :
# "facebook/blenderbot-400M-distill" pour un chatbot conversationnel
# "microsoft/DialoGPT-small" pour une version plus rapide

# ---------- Initialisation du chatbot ----------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

if 'hf_token' not in st.session_state:
    st.session_state.hf_token = None

def add_to_chat(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

# ---------- Fonction pour interroger l'API Hugging Face ----------
def query_huggingface(payload, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Modèle en cours de chargement, veuillez patienter..."}
        else:
            return {"error": f"Erreur API: {response.status_code}"}
    except Exception as e:
        return {"error": f"Erreur de connexion: {str(e)}"}

def get_medical_context():
    return """Vous êtes un assistant médical virtuel spécialisé dans le diabète. 
    Votre rôle est de fournir des informations générales sur le diabète, ses symptômes, 
    la prévention et les conseils de santé. Vous devez toujours rappeler que vos conseils 
    ne remplacent pas une consultation médicale professionnelle."""

# ---------- Données et Modèle ----------
@st.cache_data
def load_model():
    # Création de données simulées pour la démo
    np.random.seed(42)
    n_samples = 768
    
    # Génération de données réalistes
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
    
    # Création de la variable cible basée sur des règles médicales
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

# ---------- Configuration de la page ----------
st.set_page_config(
    page_title="DiabPredict - Prédiction de Diabète",
    page_icon="🏥",
    layout="wide"
)

# ---------- Logo et en-tête ----------
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <h2 style="color:#2b5876;">🏥 DiabPredict</h2>
    <hr style="border:1px solid #2b5876;">
</div>
""", unsafe_allow_html=True)

# ---------- Configuration Hugging Face ----------
st.sidebar.markdown("### 🤖 Configuration Chatbot")
hf_token = st.sidebar.text_input(
    "Token Hugging Face (optionnel)", 
    type="password",
    help="Pour éviter les limitations de rate, obtenez un token gratuit sur huggingface.co"
)
if hf_token:
    st.session_state.hf_token = hf_token

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
chat_button = st.sidebar.button("💬 Assistant Médical IA", help="Discuter avec notre assistant IA spécialisé en diabète")

if chat_button:
    st.session_state.show_chat = not st.session_state.show_chat

# ---------- Fonctionnalité du ChatBot avec LLM ----------
def medical_chatbot():
    st.markdown("""
    <style>
    .chatbox {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 400px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        z-index: 1000;
        padding: 0;
        max-height: 600px;
        overflow: hidden;
        border: 2px solid #2b5876;
    }
    .chat-header {
        background: linear-gradient(135deg, #2b5876, #4e89ae);
        color: white;
        padding: 15px;
        text-align: center;
        border-radius: 13px 13px 0 0;
    }
    .chat-content {
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 10px 15px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0;
        max-width: 85%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: linear-gradient(135deg, #f5f5f5, #e0e0e0);
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 0;
        max-width: 85%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 3px solid #2b5876;
    }
    .loading {
        color: #2b5876;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.show_chat:
        st.markdown(f"""
        <div class="chatbox">
            <div class="chat-header">
                <h3>🤖 Assistant Diabète IA</h3>
                <small>Propulsé par Hugging Face</small>
            </div>
            <div class="chat-content">
        """, unsafe_allow_html=True)

        # Afficher l'historique du chat
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {chat["message"]}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Zone de saisie
        user_input = st.text_input(
            "Posez votre question sur le diabète...", 
            key="chat_input", 
            label_visibility="collapsed",
            placeholder="Ex: Quels sont les symptômes du diabète ?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            send_button = st.button("Envoyer", type="primary")
        
        if send_button and user_input:
            add_to_chat("user", user_input)
            
            # Afficher un message de chargement
            with st.spinner("🤖 L'IA réfléchit..."):
                # Préparer le contexte médical
                medical_prompt = f"{get_medical_context()}\n\nQuestion du patient: {user_input}\n\nRéponse:"
                
                payload = {
                    "inputs": medical_prompt,
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.7,
                        "do_sample": True,
                        "top_p": 0.9
                    }
                }
                
                # Interroger l'API Hugging Face
                result = query_huggingface(payload, st.session_state.hf_token)
                
                if "error" in result:
                    if "loading" in result["error"].lower():
                        response = "🔄 Le modèle IA se charge, veuillez patienter quelques instants et réessayer."
                    else:
                        # Réponses de fallback en cas d'erreur
                        fallback_responses = {
                            "symptômes": "Les symptômes courants du diabète incluent : soif excessive, mictions fréquentes, fatigue inexpliquée, vision floue, cicatrisation lente des plaies, et perte de poids inexpliquée. Si vous ressentez ces symptômes, consultez un médecin.",
                            "prévention": "Pour prévenir le diabète de type 2 : 1) Maintenez un poids santé 2) Pratiquez une activité physique régulière (150 min/semaine) 3) Adoptez une alimentation équilibrée riche en fibres 4) Évitez les sucres ajoutés 5) Ne fumez pas 6) Limitez l'alcool.",
                            "alimentation": "Privilégiez : légumes verts, céréales complètes, légumineuses, protéines maigres, poissons gras, noix. Évitez : sucres raffinés, boissons sucrées, aliments ultra-transformés, graisses trans.",
                            "urgence": "⚠️ Signes d'urgence diabétique : confusion, perte de conscience, respiration rapide, nausées/vomissements, douleurs abdominales. Contactez immédiatement le 15 (SAMU).",
                            "diagnostic": "Le diagnostic se fait par tests sanguins : glycémie à jeun (≥126 mg/dL), HbA1c (≥6.5%), ou test d'hyperglycémie provoquée. Consultez votre médecin traitant pour un dépistage."
                        }
                        
                        response = "Je rencontre des difficultés techniques avec l'IA. Voici une réponse basée sur des connaissances médicales validées :"
                        for key, value in fallback_responses.items():
                            if key in user_input.lower():
                                response += f"\n\n{value}"
                                break
                        else:
                            response += "\n\nPour des conseils personnalisés sur le diabète, je recommande de consulter un professionnel de santé qualifié."
                else:
                    # Extraire la réponse générée
                    if isinstance(result, list) and len(result) > 0:
                        response = result[0].get("generated_text", "").replace(medical_prompt, "").strip()
                    elif isinstance(result, dict):
                        response = result.get("generated_text", "").replace(medical_prompt, "").strip()
                    else:
                        response = "Je n'ai pas pu générer une réponse appropriée. Veuillez reformuler votre question."
                    
                    # Nettoyer et valider la réponse
                    if not response or len(response) < 10:
                        response = "Je vous recommande de consulter un professionnel de santé pour des conseils personnalisés sur le diabète."
                    
                    # Ajouter un disclaimer médical
                    response += "\n\n⚠️ Cette information est à titre éducatif uniquement et ne remplace pas un avis médical professionnel."
                
                add_to_chat("bot", response)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

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

# ---------- Page 5 : À propos ----------
elif choice == "ℹ️ À propos":
    st.title("ℹ️ À Propos de DiabPredict")
    st.markdown("""
    ## 🏥 DiabPredict - Assistant Médical IA
    
    Cette application utilise l'intelligence artificielle pour :
    - **Prédire le risque de diabète** basé sur vos paramètres médicaux
    - **Analyser les facteurs de risque** les plus importants
    - **Simuler l'impact** de différents paramètres cliniques
    - **Fournir un chatbot médical** alimenté par Hugging Face
    
    ### 🤖 Chatbot IA
    Notre assistant virtuel utilise des modèles de langage avancés de Hugging Face pour répondre à vos questions sur le diabète.
    
    ### ⚠️ Avertissement Médical
    Cette application est uniquement à des fins éducatives et ne remplace pas un avis médical professionnel.
    
    ### 🔧 Technologies Utilisées
    - **Streamlit** pour l'interface utilisateur
    - **Scikit-learn** pour le machine learning
    - **Hugging Face API** pour l'IA conversationnelle
    - **Random Forest** pour la prédiction
    
    ---
    **Développé avec ❤️ pour la prévention du diabète**
    """)

# ---------- Afficher le chatbot sur toutes les pages ----------
medical_chatbot()
