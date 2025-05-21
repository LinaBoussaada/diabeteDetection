# ---------- Page 1 : Prédiction ----------
if st.session_state.active_tab == "prediction":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📋 Données du patient")
        
        # Actions rapides
        st.markdown("""
        <div class="quick-help-tabs">
            <div class="quick-help-tab" onclick="document.getElementById('predict_btn').click()">🔍 Analyse rapide</div>
            <div class="quick-help-tab" onclick="resetValues()">🔄 Réinitialiser</div>
            <div class="quick-help-tab" onclick="loadSampleData()">📊 Données d'exemple</div>
        </div>
        <script>
        function resetValues() {
            // Réinitialiser tous les champs numérique
            const inputs = document.querySelectorAll('input[type="number"]');
            inputs.forEach(input => {
                input.value = '';
                input.dispatchEvent(new Event('change', { bubbles: true }));
            });
        }
        
        function loadSampleData() {
            // Cette fonction serait mieux implémentée côté Streamlit mais voici un aperçu
            alert("Fonctionnalité à venir: Chargement de données d'exemple");
        }
        </script>
        """, unsafe_allow_html=True)
        
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
        
        field_help = {
            "Pregnancies": "Nombre de grossesses",
            "Glucose": "Taux de glucose dans le sang après 2h (test de tolérance)",
            "BloodPressure": "Pression artérielle diastolique (mm Hg)",
            "SkinThickness": "Épaisseur du pli cutané tricipital (mm)",
            "Insulin": "Taux d'insuline sérique à 2h (mu U/ml)",
            "BMI": "Indice de masse corporelle (poids en kg/(taille en m)²)",
            "DiabetesPedigreeFunction": "Score génétique basé sur les antécédents familiaux",
            "Age": "Âge en années",
            "Glucose_to_BMI": "Ratio calculé automatiquement",
            "Insulin_to_Glucose": "Ratio calculé automatiquement",
            "Is_Elderly": "1 si âge > 50 ans, 0 sinon"
        }
        
        # Première colonne de champs
        with basic_cols:
            for i, col in enumerate(columns[:6]):
                default_val = data[col].median() if col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] else 0.0
                label = field_labels.get(col, col)
                help_text = field_help.get(col, "")
                val = st.number_input(f"{label}", 
                                     value=float(default_val), 
                                     key=f"pred_{col}",
                                     help=help_text)
                input_data.append(val)
        
        # Deuxième colonne de champs
        with advanced_cols:
            for i, col in enumerate(columns[6:]):
                default_val = data[col].median() if col in data.columns else 0.0
                label = field_labels.get(col, col)
                help_text = field_help.get(col, "")
                val = st.number_input(f"{label}", 
                                     value=float(default_val), 
                                     key=f"pred_{col}",
                                     help=help_text)
                input_data.append(val)
        
        if st.button("🔍 Analyser le risque", key="predict_btn"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            result_class = "positive" if prediction == 1 else "negative"
            result_text = "Risque de diabète détecté" if prediction == 1 else "Pas de risque de diabète détecté"
            
            st.markdown(f'<div class="prediction-box {result_class}">{result_text} (Confiance: {probability:.2%})</div>', unsafe_allow_html=True)
            
            # Jauge de risque visuelle
            st.markdown(f"""
            <div style="margin: 20px 0;">
                <div style="font-weight: bold; margin-bottom: 5px;">Niveau de risque:</div>
                <div style="background-color: #f1f1f1; border-radius: 20px; height: 20px; position: relative;">
                    <div style="width: {probability*100}%; background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c); border-radius: 20px; height: 100%; position: absolute;">
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>Faible</span>
                    <span>Modéré</span>
                    <span>Élevé</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggestions en fonction du résultat
            if prediction == 1:
                st.warning("⚠️ **Recommandations:**\n"
                           "- Consultez rapidement un médecin pour un diagnostic complet\n"
                           "- Surveillez votre glycémie régulièrement\n"
                           "- Adoptez une alimentation équilibrée et faible en sucres")
                
                # Facteurs de risque identifiés
                feature_importance = model.feature_importances_
                sorted_indices = np.argsort(feature_importance)[::-1]
                top_factors = [(columns[i], input_data[i], feature_importance[i]) for i in sorted_indices[:3]]
                
                st.markdown("#### Principaux facteurs de risque identifiés:")
                for factor, value, importance in top_factors:
                    st.markdown(f"- **{field_labels.get(factor, factor)}**: {value:.1f} ({importance*100:.1f}% d'importance)")
            else:
                st.success("✅ **Continuez les bonnes habitudes:**\n"
                          "- Maintenez une activité physique régulière\n"
                          "- Adoptez une alimentation équilibrée\n"
                          "- Faites des bilans de santé annuels")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("💡 Saviez-vous?")
        
        # Onglets d'information
        info_tabs = st.tabs(["📊 Statistiques", "⚠️ Signes d'alerte", "🔬 Comprendre le test"])
        
        with info_tabs[0]:
            st.info("""
            **Le diabète en chiffres:**
            - Plus de 420 millions de personnes dans le monde vivent avec le diabète
            - 1 personne sur 2 atteintes de diabète l'ignore
            - 1 adulte sur 10 pourrait développer un diabète d'ici 2045
            
            **En France:**
            - Plus de 3,5 millions de personnes traitées pour le diabète
            - 90% des cas sont des diabètes de type 2
            - Environ 700 000 personnes diabétiques ignorent leur pathologie
            """)
            
            # Mini graphique des tendances
            st.markdown("##### Tendance des diagnostics de diabète")
            chart_data = pd.DataFrame({
                'Année': [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
                'Millions de cas': [2.9, 3.0, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8]
            })
            st.line_chart(chart_data.set_index('Année'))
            
        with info_tabs[1]:
            st.warning("""
            **Signes d'alerte à surveiller:**
            - Soif excessive et bouche sèche
            - Mictions fréquentes, surtout la nuit
            - Fatigue inexpliquée
            - Vision floue
            - Plaies qui cicatrisent lentement
            - Infections cutanées récurrentes
            - Fourmillements ou engourdissements
            
            **Quand consulter immédiatement:**
            Si vous ressentez une soif extrême, une fatigue sévère ou une perte de poids rapide, consultez rapidement un médecin.
            """)
            
        with info_tabs[2]:
            st.markdown("""
            #### Comment interpréter les valeurs?
            
            📊 **Glycémie à jeun:**
            - **Normal:** < 100 mg/dL
            - **Prédiabète:** 100-125 mg/dL
            - **Diabète:** ≥ 126 mg/dL
            
            📈 **Test de tolérance au glucose (2h):**
            - **Normal:** < 140 mg/dL
            - **Prédiabète:** 140-199 mg/dL
            - **Diabète:** ≥ 200 mg/dL
            
            🔍 **HbA1c (hémoglobine glyquée):**
            - **Normal:** < 5,7%
            - **Prédiabète:** 5,7% à 6,4% 
            - **Diabète:** ≥ 6,5%
            """)
        
        # Ajouter un lien rapide vers les tests disponibles
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top:0">📱 Suivi de votre diabète</h4>
            <p>Téléchargez notre application mobile DiabTrack pour suivre votre glycémie quotidiennement.</p>
            <div style="display: flex; gap: 10px;">
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
    layout="wide",
    initial_sidebar_state="collapsed"  # Masquer la barre latérale par défaut
)

# Fonction pour ajouter le logo
def add_logo():
    try:
        with open("logo.png", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            logo_html = f'''
                <div style="display: flex; justify-content: center; margin-bottom: 10px;">
                    <img src="data:image/png;base64,{b64}" width="150px" alt="Logo DiabPredict">
                </div>
            '''
            st.markdown(logo_html, unsafe_allow_html=True)
    except FileNotFoundError:
        # Logo par défaut au format SVG si le fichier logo.png n'est pas trouvé
        logo_svg = '''
        <svg width="150" height="80" viewBox="0 0 150 80">
            <rect x="20" y="15" width="110" height="50" rx="10" fill="#3498db" />
            <circle cx="50" cy="40" r="15" fill="#e74c3c" />
            <polygon points="75,25 90,40 75,55 60,40" fill="#2ecc71" />
            <text x="105" y="45" font-family="Arial" font-size="18" font-weight="bold" fill="white">DiabPredict</text>
        </svg>
        '''
        st.markdown(f'<div style="display: flex; justify-content: center; margin-bottom: 10px;">{logo_svg}</div>', unsafe_allow_html=True)

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
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
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
    
    /* Navigation horizontale moderne */
    .nav-bar {
        background-color: white;
        padding: 10px;
        border-radius: 50px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        margin: 0 auto 25px auto;
        display: flex;
        justify-content: space-between;
        max-width: 95%;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: none;
    }
    
    .nav-bar::-webkit-scrollbar {
        display: none;
    }
    
    .nav-item {
        background-color: transparent;
        padding: 10px 20px;
        border-radius: 30px;
        cursor: pointer;
        font-weight: 500;
        text-align: center;
        transition: all 0.3s;
        color: var(--dark);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .nav-item:hover {
        background-color: var(--neutral);
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
        border-radius: 8px;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        transition: all 0.2s;
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
        border-radius: 8px;
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
    
    /* Onglets d'aide rapide */
    .quick-help-tabs {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 5px;
        margin: 10px 0;
    }
    
    .quick-help-tab {
        background-color: var(--neutral);
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-help-tab:hover {
        background-color: var(--primary);
        color: white;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
    }
    
    .badge-primary {
        background-color: var(--primary);
        color: white;
    }
    
    .badge-secondary {
        background-color: var(--secondary);
        color: white;
    }
    
    .badge-warning {
        background-color: var(--warning);
        color: white;
    }
    
    /* Progress bar personnalisée */
    .stProgress > div > div > div {
        background-color: var(--primary);
    }
    
    /* Améliorations responsives */
    @media (max-width: 768px) {
        .nav-item {
            padding: 8px 12px;
            font-size: 14px;
        }
        
        .main-header h1 {
            font-size: 24px;
        }
    }
    
    /* Rendre les inputs plus élégants */
    div[data-baseweb="input"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Menu fixe en haut - barre secondaire */
    .secondary-nav {
        position: sticky;
        top: 0;
        background-color: rgba(255, 255, 255, 0.95);
        z-index: 100;
        padding: 10px 0;
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .secondary-nav-links {
        display: flex;
        gap: 15px;
    }
    
    .secondary-nav-link {
        color: var(--dark);
        text-decoration: none;
        font-weight: 500;
        font-size: 14px;
    }
    
    .secondary-nav-link:hover {
        color: var(--primary);
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

# Barre de navigation secondaire fixe
st.markdown("""
<div class="secondary-nav">
    <div>
        <img src="https://img.icons8.com/color/48/000000/diabetes.png" width="24px" style="vertical-align: middle;"/> 
        <b style="vertical-align: middle;">DiabPredict</b>
    </div>
    <div class="secondary-nav-links">
        <a href="https://www.federationdesdiabetiques.org/" target="_blank" class="secondary-nav-link">Ressources</a>
        <a href="#" class="secondary-nav-link">Aide</a>
        <a href="#footer" class="secondary-nav-link">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>💉 DiabPredict - Plateforme de prédiction du diabète</h1></div>', unsafe_allow_html=True)

# Navigation moderne avec icônes
st.markdown("""
<div class="nav-bar">
    <div class="nav-item" id="prediction" onclick="handleNavClick('prediction')">
        <i class="fas fa-home"></i> 🏠 Prédiction
    </div>
    <div class="nav-item" id="explanation" onclick="handleNavClick('explanation')">
        <i class="fas fa-chart-bar"></i> 📈 Explication
    </div>
    <div class="nav-item" id="simulation" onclick="handleNavClick('simulation')">
        <i class="fas fa-sync"></i> 🔁 Simulation
    </div>
    <div class="nav-item" id="history" onclick="handleNavClick('history')">
        <i class="fas fa-history"></i> 🗂️ Historique
    </div>
    <div class="nav-item" id="chatbot" onclick="handleNavClick('chatbot')">
        <i class="fas fa-comments"></i> 💬 ChatBot
    </div>
</div>

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
        
        // Mettre à jour l'URL
        window.history.pushState({}, '', '?tab=' + tabName);
    }
    
    // Initialiser la navigation active basée sur l'URL
    document.addEventListener('DOMContentLoaded', function() {
        const urlParams = new URLSearchParams(window.location.search);
        const activeTab = urlParams.get('tab') || 'prediction';
        document.getElementById(activeTab).classList.add('nav-active');
    });
</script>
""", unsafe_allow_html=True)

# Obtenir l'onglet actif (par défaut: prédiction)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "prediction"

# Utiliser st.query_params au lieu de st.query_params
tab = st.query_params.get("tab", ["prediction"])[0] if isinstance(st.query_params.get("tab"), list) else st.query_params.get("tab", "prediction")
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
    📞 Téléphone: +216 123456789
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
    **Version:** 1.0
    **Dernière mise à jour:** Mai 2025
    
    **Nouveautés:**
    - Interface améliorée
    - Navigation horizontale
    - Assistant éducatif
    - Visualisations interactives
    """)

st.markdown('</div>', unsafe_allow_html=True)
