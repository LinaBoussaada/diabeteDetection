import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------- Configuration de la page et style ----------
st.set_page_config(
    page_title="DiabPredict - Prédiction Diabète",
    page_icon="🩺",
    layout="wide"
)

# Fonction pour ajouter le logo
def add_logo():
    try:
        with open("logo.png", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            logo_html = f'''
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{b64}" width="180px" alt="Logo DiabPredict">
                </div>
            '''
            st.markdown(logo_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("📌 Logo non trouvé. Placez 'logo.png' dans le répertoire principal.")

# CSS personnalisé
st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --warning: #e74c3c;
        --neutral: #ecf0f1;
        --dark: #2c3e50;
    }
    
    /* En-tête */
    .main-header {
        background-color: var(--neutral);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Cartes */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    
    .nav-item {
        background-color: var(--neutral);
        padding: 10px 15px;
        border-radius: 20px;
        cursor: pointer;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s;
    }
    
    .nav-item:hover {
        background-color: var(--primary);
        color: white;
    }
    
    .nav-active {
        background-color: var(--primary);
        color: white;
    }
    
    /* Boutons */
    .stButton>button {
        width: 100%;
        background-color: var(--primary);
        color: white;
        border: none;
        font-weight: bold;
        padding: 10px 15px;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
    }
    
    /* Pied de page */
    .footer {
        background-color: var(--dark);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        margin-top: 30px;
    }
    
    /* Résultats */
    .prediction-box {
        text-align: center;
        font-size: 24px;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .positive {
        background-color: #ffcdd2;
        color: #c62828;
    }
    
    .negative {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

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

# ---------- En-tête avec logo ----------
add_logo()
st.markdown('<div class="main-header"><h1>💉 DiabPredict - Plateforme de prédiction du diabète</h1></div>', unsafe_allow_html=True)

# ---------- Navigation horizontale ----------
tabs = {
    "prediction": "🏠 Prédiction",
    "explanation": "📈 Explication",
    "simulation": "🔁 Simulation",
    "history": "🗂️ Historique",
    "chatbot": "💬 ChatBot"
}

# Créer la navigation horizontale
nav_html = '<div class="nav-container">'
for key, value in tabs.items():
    nav_html += f'<div class="nav-item" id="{key}" onclick="handleNavClick(\'{key}\')">{value}</div>'
nav_html += '</div>'

# JavaScript pour gérer la navigation
nav_js = """
<script>
    function handleNavClick(tabName) {
        // Envoyer l'événement de changement d'onglet à Streamlit
        const event = new CustomEvent('streamlit:setComponentValue', {
            detail: { value: tabName }
        });
        window.dispatchEvent(event);
        
        // Mettre à jour visuellement l'onglet actif
        document.querySelectorAll('.nav-item').forEach(item => {
            if (item.id === tabName) {
                item.classList.add('nav-active');
            } else {
                item.classList.remove('nav-active');
            }
        });
    }
    
    // Initialiser
    document.addEventListener('DOMContentLoaded', function() {
        // Activer l'onglet par défaut
        const defaultTab = 'prediction';
        document.getElementById(defaultTab).classList.add('nav-active');
    });
</script>
"""

# Afficher la navigation
st.components.v1.html(nav_html + nav_js, height=80)

# Obtenir l'onglet actif (par défaut: prédiction)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "prediction"

tab = st.experimental_get_query_params().get("tab", ["prediction"])[0]
st.session_state.active_tab = tab

# ---------- Page 1 : Prédiction ----------
if st.session_state.active_tab == "prediction":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📋 Données du patient")
        
        # Diviser les entrées en deux colonnes pour économiser de l'espace
        basic_cols, advanced_cols = st.columns(2)
        
        input_data = []
        field_labels = {
            "Pregnancies": "Grossesses",
            "Glucose": "Glucose (mg/dL)",
            "BloodPressure": "Pression artérielle (mmHg)",
            "SkinThickness": "Épaisseur de peau (mm)",
            "Insulin": "Insuline (µU/mL)",
            "BMI": "IMC",
            "DiabetesPedigreeFunction": "Fonction pedigree diabétique",
            "Age": "Âge",
            "Glucose_to_BMI": "Ratio Glucose/IMC",
            "Insulin_to_Glucose": "Ratio Insuline/Glucose",
            "Is_Elderly": "Patient âgé (>50 ans)"
        }
        
        # Première colonne de champs
        with basic_cols:
            for i, col in enumerate(columns[:6]):
                default_val = data[col].median() if col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] else 0.0
                label = field_labels.get(col, col)
                val = st.number_input(f"{label}", value=float(default_val), key=f"pred_{col}")
                input_data.append(val)
        
        # Deuxième colonne de champs
        with advanced_cols:
            for i, col in enumerate(columns[6:]):
                default_val = data[col].median() if col in data.columns else 0.0
                label = field_labels.get(col, col)
                val = st.number_input(f"{label}", value=float(default_val), key=f"pred_{col}")
                input_data.append(val)
        
        if st.button("🔍 Analyser le risque", key="predict_btn"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            result_class = "positive" if prediction == 1 else "negative"
            result_text = "Risque de diabète détecté" if prediction == 1 else "Pas de risque de diabète détecté"
            
            st.markdown(f'<div class="prediction-box {result_class}">{result_text} (Confiance: {probability:.2%})</div>', unsafe_allow_html=True)
            
            # Suggestions en fonction du résultat
            if prediction == 1:
                st.warning("⚠️ **Recommandations:**\n"
                           "- Consultez rapidement un médecin pour un diagnostic complet\n"
                           "- Surveillez votre glycémie régulièrement\n"
                           "- Adoptez une alimentation équilibrée et faible en sucres")
            else:
                st.success("✅ **Continuez les bonnes habitudes:**\n"
                          "- Maintenez une activité physique régulière\n"
                          "- Adoptez une alimentation équilibrée\n"
                          "- Faites des bilans de santé annuels")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("💡 Saviez-vous?")
        st.info("""
        **Le diabète en chiffres:**
        - Plus de 420 millions de personnes dans le monde vivent avec le diabète
        - 1 personne sur 2 atteintes de diabète l'ignore
        - 1 adulte sur 10 pourrait développer un diabète d'ici 2045
        
        **Signes d'alerte:**
        - Soif excessive et bouche sèche
        - Mictions fréquentes
        - Fatigue inexpliquée
        - Vision floue
        - Plaies qui cicatrisent lentement
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📊 Statistiques principales")
        values = {}
        for col in ["Glucose", "BMI", "Age", "BloodPressure", "Insulin"]:
            if col in columns:
                values[col] = data[col].median()
        
        stats_df = pd.DataFrame({"Facteur": list(values.keys()), "Valeur médiane": list(values.values())})
        st.dataframe(stats_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page 2 : Explication du modèle ----------
elif st.session_state.active_tab == "explanation":
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📊 Explication du Modèle Random Forest")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Importance des caractéristiques
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Caractéristique': columns, 'Importance (%)': importances * 100})
        feature_df = feature_df.sort_values(by='Importance (%)', ascending=False)
        st.bar_chart(feature_df.set_index('Caractéristique'))
    
    with col2:
        st.markdown("""
        ### 🔎 Comment fonctionne le modèle?
        
        Notre modèle d'**intelligence artificielle** utilise la technique des **forêts aléatoires** pour prédire le risque de diabète. Cette méthode:
        
        - Combine plusieurs arbres de décision
        - Analyse l'importance de chaque facteur
        - Génère une prédiction avec une confiance exprimée en pourcentage
        
        Les facteurs les plus importants sont généralement:
        - Le taux de glucose
        - L'IMC (Indice de Masse Corporelle)
        - La fonction pedigree du diabète (facteur héréditaire)
        - L'âge du patient
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section supplémentaire expliquant les facteurs
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📝 Explication des facteurs principaux")
    
    factors = {
        "Glucose": "Mesure du taux de glucose dans le sang. Une valeur élevée (>140 mg/dL) peut indiquer un diabète.",
        "BMI": "Indice de Masse Corporelle. Un IMC > 30 indique une obésité, facteur de risque majeur.",
        "Age": "Le risque de diabète de type 2 augmente avec l'âge, particulièrement après 45 ans.",
        "Pregnancies": "Nombre de grossesses. Chaque grossesse augmente légèrement le risque de diabète.",
        "DiabetesPedigreeFunction": "Mesure du risque génétique basée sur les antécédents familiaux.",
        "BloodPressure": "Pression artérielle. L'hypertension est souvent associée au diabète.",
        "Insulin": "Niveau d'insuline à jeun. Un niveau anormal indique des problèmes de régulation.",
        "SkinThickness": "Mesure indirecte de la graisse corporelle, liée au risque de diabète."
    }
    
    for factor, description in factors.items():
        st.markdown(f"**{factor}**: {description}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page 3 : Simulation de scénario ----------
elif st.session_state.active_tab == "simulation":
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("🔍 Simulation : impact d'une variable sur le risque")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_feature = st.selectbox("Choisir une variable à modifier", columns)
        
        # Valeurs min et max de la caractéristique
        feature_min = max(0, data[selected_feature].min())
        feature_max = min(200, data[selected_feature].max() * 1.5)
        
        # Valeur par défaut
        default_value = data[selected_feature].median()
        
        values = st.slider(f"Valeur de {selected_feature}", 
                          float(feature_min), 
                          float(feature_max), 
                          float(default_value), 
                          step=float((feature_max-feature_min)/100))
        
        # Créer un ensemble d'entrées basées sur les médianes
        base_input = [data[col].median() if col in data.columns else 0.0 for col in columns]
        
        # Remplacer la valeur de la caractéristique sélectionnée
        base_input[list(columns).index(selected_feature)] = values
        
        input_array = np.array(base_input).reshape(1, -1)
        prediction = model.predict(scaler.transform(input_array))[0]
        probability = model.predict_proba(scaler.transform(input_array))[0][1]
        
        result_class = "positive" if prediction == 1 else "negative"
        result_text = "Risque de diabète" if prediction == 1 else "Pas de risque de diabète"
        
        st.markdown(f'<div class="prediction-box {result_class}">{result_text} (Probabilité: {probability:.2%})</div>', unsafe_allow_html=True)
    
    with col2:
        # Simuler différentes valeurs pour la caractéristique sélectionnée
        step = (feature_max - feature_min) / 10
        test_values = np.arange(feature_min, feature_max, step)
        probabilities = []
        
        for val in test_values:
            test_input = base_input.copy()
            test_input[list(columns).index(selected_feature)] = val
            test_array = np.array(test_input).reshape(1, -1)
            prob = model.predict_proba(scaler.transform(test_array))[0][1]
            probabilities.append(prob * 100)  # Convertir en pourcentage
        
        # Créer un DataFrame pour le graphique
        sim_df = pd.DataFrame({
            selected_feature: test_values,
            'Risque (%)': probabilities
        })
        
        st.line_chart(sim_df.set_index(selected_feature))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ajouter plus d'informations contextuelles
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📖 Comprendre cette simulation")
    st.markdown("""
    Cette simulation vous permet de comprendre comment chaque facteur influence le risque de diabète.
    
    - **Ligne bleue** : montre comment le risque de diabète évolue lorsque vous modifiez une seule variable
    - **Toutes les autres variables** sont maintenues à leur valeur médiane
    
    ### Comment utiliser cette simulation:
    1. Sélectionnez un facteur dans le menu déroulant
    2. Faites glisser le curseur pour voir l'impact sur le risque
    3. Observez comment la probabilité change sur le graphique
    
    Cette fonctionnalité est particulièrement utile pour identifier les facteurs sur lesquels vous devriez vous concentrer pour réduire votre risque.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page 4 : Historique personnel ----------
elif st.session_state.active_tab == "history":
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("🧾 Journal de suivi personnel")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📝 Entrer une nouvelle mesure")
        
        # Divisez les entrées en plusieurs colonnes pour gagner de l'espace
        cols = st.columns(3)
        
        input_data = []
        field_labels = {
            "Pregnancies": "Grossesses",
            "Glucose": "Glucose (mg/dL)",
            "BloodPressure": "Pression (mmHg)",
            "SkinThickness": "Peau (mm)",
            "Insulin": "Insuline (µU/mL)",
            "BMI": "IMC",
            "DiabetesPedigreeFunction": "Fonction pedigree",
            "Age": "Âge",
            "Glucose_to_BMI": "Glucose/IMC",
            "Insulin_to_Glucose": "Insuline/Glucose",
            "Is_Elderly": "Patient âgé"
        }
        
        # Distribuer les champs dans les colonnes
        for i, col in enumerate(columns):
            column_index = i % 3
            with cols[column_index]:
                default_val = data[col].median() if col in data.columns else 0.0
                label = field_labels.get(col, col)
                val = st.number_input(f"{label}", value=float(default_val), key=f"hist_{col}")
                input_data.append(val)
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            date = st.date_input("Date de la mesure", key="measure_date")
        with date_col2:
            notes = st.text_input("Notes (optionnel)", key="measure_notes")
        
        if st.button("📅 Ajouter au journal", key="add_to_history"):
            input_array = np.array(input_data).reshape(1, -1)
            pred = model.predict(scaler.transform(input_array))[0]
            prob = model.predict_proba(scaler.transform(input_array))[0][1]
            result = "À risque" if pred == 1 else "Normal"
            
            st.session_state.history.append({
                "date": date.strftime("%d/%m/%Y"),
                "data": input_data,
                "result": result,
                "probability": f"{prob:.2%}",
                "notes": notes
            })
            
            st.success("✅ Mesure ajoutée au journal!")
    
    with col2:
        st.markdown("### 📊 Statistiques")
        if st.session_state.history:
            risk_count = sum(1 for entry in st.session_state.history if entry["result"] == "À risque")
            normal_count = len(st.session_state.history) - risk_count
            risk_percent = (risk_count / len(st.session_state.history)) * 100
            
            st.metric("Total des entrées", len(st.session_state.history))
            st.metric("Résultats à risque", f"{risk_count} ({risk_percent:.1f}%)")
            st.metric("Résultats normaux", normal_count)
    
    # Afficher l'historique
    st.markdown("### 📜 Historique des mesures")
    if st.session_state.history:
        # Créer un DataFrame pour l'affichage
        history_data = []
        for entry in st.session_state.history:
            row = {"Date": entry["date"], "Résultat": entry["result"], "Probabilité": entry["probability"]}
            if entry["notes"]:
                row["Notes"] = entry["notes"]
            for i, col in enumerate(columns):
                label = field_labels.get(col, col)
                row[label] = entry["data"][i]
            history_data.append(row)
        
        df = pd.DataFrame(history_data)
        st.dataframe(df)
        
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.success("Historique effacé!")
    else:
        st.info("🔍 Aucune mesure enregistrée. Ajoutez votre première mesure ci-dessus.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page 5 : ChatBot éducatif ----------
elif st.session_state.active_tab == "chatbot":
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("💬 Assistant diabète - Réponses aux questions fréquentes")
    
    # FAQ améliorée
    faq = {
        "Quels sont les symptômes du diabète ?": 
            """
            Les symptômes courants du diabète incluent:
            - Soif excessive et bouche sèche
            - Mictions fréquentes (surtout la nuit)
            - Fatigue inexpliquée et irritabilité
            - Vision floue
            - Plaies qui cicatrisent lentement
            - Infections fréquentes (peau, gencives, vessie)
            - Faim excessive
            - Perte de poids inexpliquée (type 1)
            - Picotements ou engourdissements des mains/pieds (neuropathie)
            
            Il est important de consulter un médecin si vous présentez ces symptômes.
            """,
            
        "Quelle est la différence entre diabète type 1 et type 2 ?":
            """
            **Diabète de type 1:**
            - Maladie auto-immune où le pancréas ne produit pas d'insuline
            - Apparaît généralement pendant l'enfance ou l'adolescence
            - Nécessite des injections d'insuline à vie
            - Non lié au mode de vie ou au poids
            - Représente environ 10% des cas de diabète
            
            **Diabète de type 2:**
            - Le corps devient résistant à l'insuline ou n'en produit pas suffisamment
            - Apparaît généralement chez l'adulte (mais de plus en plus chez les jeunes)
            - Souvent lié au surpoids, à la sédentarité et aux facteurs génétiques
            - Peut être contrôlé par l'alimentation, l'exercice et des médicaments
            - Représente environ 90% des cas de diabète
            """,
            
        "Comment prévenir le diabète ?":
            """
            Pour prévenir le diabète de type 2:
            
            **Alimentation saine:**
            - Privilégiez les légumes, fruits, céréales complètes
            - Limitez les sucres ajoutés et les glucides raffinés
            - Choisissez des graisses saines (huile d'olive, noix, poissons)
            
            **Activité physique:**
            - Pratiquez au moins 150 minutes d'exercice modéré par semaine
            - Intégrez de la marche quotidienne
            - Évitez les longues périodes d'inactivité
            
            **Poids santé:**
            - Maintenez un IMC entre 18,5 et 24,9
            - Même une perte de poids modérée (5-7%) réduit considérablement le risque
            
            **Autres facteurs:**
            - Ne fumez pas
            - Limitez l'alcool
            - Gérez votre stress
            - Dormez suffisamment (7-8h)
            - Faites des contrôles glycémiques réguliers, surtout si vous avez des antécédents familiaux
            """,
            
        "Quel est un taux de glycémie normal ?":
            """
            **Taux de glycémie normaux:**
            
            **À jeun (sans manger depuis 8h):**
            - Normal: entre 70-99 mg/dL (3,9-5,5 mmol/L)
            - Prédiabète: entre 100-125 mg/dL (5,6-6,9 mmol/L)
            - Diabète: 126 mg/dL (7,0 mmol/L) ou plus
            
            **2 heures après un repas:**
            - Normal: moins de 140 mg/dL (7,8 mmol/L)
            - Prédiabète: entre 140-199 mg/dL (7,8-11,0 mmol/L)
            - Diabète: 200 mg/dL (11,1 mmol/L) ou plus
            
            **Test HbA1c (hémoglobine glyquée):**
            - Normal: moins de 5,7%
            - Prédiabète: entre 5,7% et 6,4%
            - Diabète: 6,5% ou plus
            
            Ces valeurs peuvent varier légèrement selon les laboratoires et les pays.
            """,
            
        "Quels aliments éviter avec le diabète ?":
            """
            **Aliments à limiter ou éviter:**
            
            **Sucres raffinés:**
            - Sodas et boissons sucrées
            - Bonbons, chocolats, pâtisseries
            - Confitures et miels en grande quantité
            
            **Glucides raffinés:**
            - Pain blanc, riz blanc
            - Pâtes non complètes
            - Céréales sucrées
            
            **Aliments transformés:**
            - Fast-food et plats préparés
            - Charcuteries
            - Snacks industriels
            
            **Graisses saturées:**
            - Viandes grasses
            - Produits laitiers entiers
            - Huiles de palme et de coco
            
            **Autres:**
            - Alcool (contient beaucoup de sucres)
            - Jus de fruits (même 100% naturels, ils sont riches en sucres)
            - Fruits secs en grande quantité
            
            Il est recommandé de consulter un nutritionniste pour un plan alimentaire personnalisé.
            """,
            
        "Comment utiliser un glucomètre ?":
            """
            **Guide d'utilisation d'un glucomètre:**
            
            1. **Préparation:**
               - Lavez et séchez vos mains
               - Préparez le lecteur, une bandelette et le dispositif de piqûre
               
            2. **Mesure:**
               - Insérez la bandelette dans le lecteur
               - Piquez le côté de votre doigt (moins douloureux que le bout)
               - Appliquez doucement la goutte de sang sur la bandelette
               - Attendez le résultat (quelques secondes)
               
            3. **Interprétation:**
               - Notez le résultat dans un carnet ou une application
               - Identifiez les tendances avec votre médecin
               
            4. **Fréquence recommandée:**
               - Variable selon votre situation (1 à plusieurs fois par jour)
               - Suivez les recommandations de votre médecin
               
            5. **Entretien:**
               - Nettoyez régulièrement le lecteur
               - Vérifiez les dates d'expiration des bandelettes
            """
    }
    
    # Interface plus interactive
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Permettre à l'utilisateur de poser une question libre ou de choisir dans la liste
        question_type = st.radio("Comment souhaitez-vous poser votre question ?", 
                                ["Choisir une question fréquente", "Poser ma propre question"])
        
        if question_type == "Choisir une question fréquente":
            question = st.selectbox("Sélectionnez votre question :", list(faq.keys()))
            
            if st.button("📚 Obtenir la réponse"):
                st.markdown(f"""
                ### 🩺 {question}
                
                {faq[question]}
                """)
        else:
            user_question = st.text_input("Tapez votre question sur le diabète :", 
                                         placeholder="Ex: Quels sont les facteurs de risque du diabète ?")
            
            if st.button("🔍 Rechercher une réponse") and user_question:
                # Recherche simple par mots-clés
                best_match = None
                max_score = 0
                
                # Mots-clés pour chaque question du FAQ
                keywords = {
                    "Quels sont les symptômes du diabète ?": ["symptômes", "signes", "indication", "manifester", "symptome"],
                    "Quelle est la différence entre diabète type 1 et type 2 ?": ["différence", "type", "type 1", "type 2", "distinction", "catégorie"],
                    "Comment prévenir le diabète ?": ["prévenir", "prévention", "éviter", "empêcher", "risque", "réduire"],
                    "Quel est un taux de glycémie normal ?": ["normal", "glycémie", "taux", "sucre", "sang", "niveau"],
                    "Quels aliments éviter avec le diabète ?": ["aliment", "nourriture", "manger", "éviter", "diet", "régime", "nutrition"],
                    "Comment utiliser un glucomètre ?": ["glucomètre", "mesurer", "appareil", "test", "tester", "lecteur", "glucose"]
                }
                
                # Recherche simple basée sur les mots-clés
                for q, words in keywords.items():
                    score = sum(1 for word in words if word.lower() in user_question.lower())
                    if score > max_score:
                        max_score = score
                        best_match = q
                
                if max_score > 0:
                    st.markdown(f"""
                    ### 🩺 Question similaire trouvée: {best_match}
                    
                    {faq[best_match]}
                    """)
                else:
                    st.warning("Désolé, je n'ai pas trouvé de réponse à votre question spécifique. Veuillez essayer une question dans la liste ou reformuler.")
    
    with col2:
        st.markdown("""
        ### 🔔 Bon à savoir
        
        **Quand consulter un médecin?**
        
        Consultez immédiatement si vous observez:
        - Soif extrême persistante
        - Mictions très fréquentes
        - Vision floue soudaine
        - Fatigue inexpliquée
        - Perte de poids sans raison
        
        **Ressources utiles:**
        - Association Française des Diabétiques
        - Fédération Internationale du Diabète
        - Ligne d'écoute Diabète: 0800 XX XX XX
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Pied de page (À propos) ----------
st.markdown('<div class="footer">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 📱 Contact")
    st.markdown("""
    📧 Email: ing_idl@isi.com
    🌐 Site web: www.diabpredict.org
    📞 Téléphone: +216 XX XXX XXX
    """)

with col2:
    st.markdown("### 📝 À propos de ce projet")
    st.markdown("""
    **DiabPredict** est une plateforme d'IA conçue pour aider à la prédiction du risque de diabète basée sur des données médicales.
    
    Ce projet a été réalisé avec 💡 par **Lina Boussaada & Meriem Trabelsi** dans le cadre d'un projet étudiant à l'ISI.
    
    **Technologies utilisées:** Python, Streamlit, Scikit-learn, Pandas, NumPy
    
    **Méthodes:** Random Forest, Normalisation, Feature Engineering, Visualisation de données
    
    *Cette application est destinée à un usage éducatif uniquement et ne remplace pas l'avis d'un professionnel de santé.*
    """)

with col3:
    st.markdown("### 🔄 Version")
    st.markdown("""
    **Version:** 2.0
    **Dernière mise à jour:** Mai 2025
    
    **Nouveautés:**
    - Interface améliorée
    - Navigation horizontale
    - Assistant éducatif
    - Visualisations interactives
    """)

st.markdown('</div>', unsafe_allow_html=True)
