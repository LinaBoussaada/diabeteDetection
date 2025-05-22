import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

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
    "❓ FAQ Diabète",
    "ℹ️ À propos"
])

# ---------- Page 1 : Prédiction ----------
if choice == "🩺 Prédiction":
    st.title("🩺 Prédiction du Risque Diabétique")
    st.markdown("""
    <style>
    .stNumberInput>label {
        font-size: 14px;
        color: #2b5876;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
            st.markdown("""
            **Recommandations immédiates :**
            - Consultez un médecin rapidement
            - Surveillez votre glycémie régulièrement
            - Adoptez une alimentation équilibrée
            """)
        else:
            st.success("✅ Risque faible de diabète")
            st.markdown("""
            **Conseils de prévention :**
            - Maintenez une activité physique régulière
            - Faites des bilans sanguins annuels
            - Limitez les sucres raffinés
            """)

# ---------- Page 2 : Analyse du modèle ----------
elif choice == "📊 Analyse du modèle":
    st.title("📊 Analyse du Modèle Prédictif")
    st.markdown("""
    **Méthodologie :** Random Forest avec Feature Engineering
    """)
    
    tab1, tab2 = st.tabs(["Importance des variables", "Performance"])
    
    with tab1:
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Variable': columns, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_df.set_index('Variable'))
    
    with tab2:
        st.write("Matrice de confusion (exemple sur données d'entraînement):")
        # Ici vous pourriez ajouter une vraie évaluation
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*Z54JgbS4DUwWSknhDCvNTQ.png", width=400)

# ---------- Page 3 : Simulation médicale ----------
elif choice == "🔬 Simulation médicale":
    st.title("🔬 Simulation d'Impact Clinique")
    st.info("Modifiez une variable pour voir son impact sur le risque diabétique")
    
    selected_feature = st.selectbox("Paramètre clinique", columns)
    base_value = float(data[selected_feature].median())
    min_val = float(data[selected_feature].min())
    max_val = float(data[selected_feature].max())
    
    values = st.slider(
        f"Valeur de {selected_feature}",
        min_val, max_val, base_value, 
        step=(max_val-min_val)/100,
        help="Ajustez pour voir l'impact sur la prédiction"
    )

    base_input = [data[col].median() for col in columns]
    base_input[columns.get_loc(selected_feature)] = values
    input_array = np.array(base_input).reshape(1, -1)
    prediction = model.predict(scaler.transform(input_array))[0]
    
    st.metric("Risque de diabète", "Élevé" if prediction == 1 else "Faible")
    st.progress(prediction * 100 if prediction == 1 else 30)

# ---------- Page 4 : Dossier patient ----------
elif choice == "📋 Dossier patient":
    st.title("📋 Dossier Médical Personnel")
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.subheader("Nouvelle entrée")
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
    
    if st.session_state.history:
        st.subheader("Historique des analyses")
        df = pd.DataFrame(st.session_state.history, columns=list(columns) + ["Résultat"])
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        
        if st.button("📤 Exporter les données"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Télécharger CSV", csv, "historique_diabete.csv", "text/csv")

# ---------- Page 5 : FAQ Diabète ----------
elif choice == "❓ FAQ Diabète":
    st.title("❓ Foire Aux Questions Médicales")
    
    faq = {
        "Quels sont les symptômes du diabète ?": {
            "answer": "Fatigue, soif excessive, mictions fréquentes, vision floue, cicatrisation lente.",
            "icon": "🆘"
        },
        "Comment prévenir le diabète ?": {
            "answer": "1. Alimentation équilibrée\n2. Activité physique régulière\n3. Contrôle du poids\n4. Bilan sanguin annuel",
            "icon": "🛡️"
        },
        "Quel est un taux de glycémie normal ?": {
            "answer": "À jeun : 0.70 à 1.10 g/L (3.9 à 6.1 mmol/L)\nAprès repas : < 1.40 g/L (7.8 mmol/L)",
            "icon": "📉"
        },
        "Quels aliments privilégier ?": {
            "answer": "Légumes verts, céréales complètes, poissons gras, fruits à faible IG (pommes, baies), noix.",
            "icon": "🥗"
        }
    }
    
    selected_question = st.selectbox("Sélectionnez une question", list(faq.keys()))
    
    st.markdown(f"""
    <div style="background:#f0f2f6;padding:15px;border-radius:10px;margin-top:10px;">
        <h4>{faq[selected_question]['icon']} {selected_question}</h4>
        <p>{faq[selected_question]['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Page 6 : À propos ----------
elif choice == "ℹ️ À propos":
    st.title("ℹ️ À Propos de DiabPredict")
    
    st.image("https://img.freepik.com/vecteurs-libre/conception-logo-hopital_23-2149610211.jpg", width=150)
    
    st.markdown("""
    <div style="background:#f0f2f6;padding:20px;border-radius:10px;">
        <h3 style="color:#2b5876;">Notre Mission</h3>
        <p>Fournir un outil prédictif accessible pour l'évaluation précoce du risque diabétique.</p>
        
        <h3 style="color:#2b5876;margin-top:20px;">Fonctionnalités Clés</h3>
        <ul>
            <li>Analyse basée sur l'IA (Random Forest)</li>
            <li>Simulation d'impact des facteurs de risque</li>
            <li>Suivi personnel des analyses</li>
            <li>Ressources éducatives sur le diabète</li>
        </ul>
        
        <h3 style="color:#2b5876;margin-top:20px;">Avertissement</h3>
        <p style="color:red;font-style:italic;">
        Cet outil ne remplace pas un diagnostic médical professionnel. 
        Consultez toujours un médecin pour une évaluation complète.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("© 2023 DiabPredict - Tous droits réservés")
