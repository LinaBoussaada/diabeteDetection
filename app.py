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

# Chargement du CSS personnalisé
def load_css():
    with open("styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except:
    # CSS de secours en cas d'erreur de chargement du fichier
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #3283C8;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour ajouter le logo
def add_logo():
    logo_html = '''
    <div class="logo-wrapper">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="20" fill="#E8F1F9"/>
            <circle cx="24" cy="16" r="8" fill="#3283C8"/>
            <path d="M16 32C16 27.582 19.582 24 24 24C28.418 24 32 27.582 32 32" stroke="#3283C8" stroke-width="4"/>
            <path d="M14 38L34 38" stroke="#3283C8" stroke-width="4" stroke-linecap="round"/>
        </svg>
        <h1>DiabPredict</h1>
    </div>
    <style>
        .logo-wrapper {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 2rem;
        }
        .logo-wrapper h1 {
            margin: 0;
            color: #3283C8;
            font-size: 2rem;
        }
    </style>
    '''
    st.markdown(logo_html, unsafe_allow_html=True)

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

# Chargement du modèle et des données
try:
    model, scaler, columns, data = load_model()
except:
    st.error("Erreur lors du chargement des données ou du modèle. Veuillez vérifier les fichiers requis.")
    model, scaler, columns, data = None, None, [], pd.DataFrame()

# ---------- En-tête et Navigation ----------
def render_header():
    header_html = '''
    <header class="header-container">
        <div class="logo-container">
            <svg class="logo-image" xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 48 48">
                <circle cx="24" cy="24" r="20" fill="#E8F1F9"/>
                <circle cx="24" cy="16" r="8" fill="#3283C8"/>
                <path d="M16 32C16 27.582 19.582 24 24 24C28.418 24 32 27.582 32 32" stroke="#3283C8" stroke-width="4"/>
                <path d="M14 38L34 38" stroke="#3283C8" stroke-width="4" stroke-linecap="round"/>
            </svg>
            <h1 class="app-title">DiabPredict</h1>
        </div>
        <nav class="nav-container">
            <a href="?tab=prediction" class="nav-item {active_prediction}">🏠 Prédiction</a>
            <a href="?tab=explanation" class="nav-item {active_explanation}">📈 Explication</a>
            <a href="?tab=simulation" class="nav-item {active_simulation}">🔁 Simulation</a>
            <a href="?tab=history" class="nav-item {active_history}">🗂️ Historique</a>
            <a href="?tab=chatbot" class="nav-item {active_chatbot}">💬 ChatBot</a>
        </nav>
    </header>
    '''.format(
        active_prediction='active' if st.session_state.active_tab == 'prediction' else '',
        active_explanation='active' if st.session_state.active_tab == 'explanation' else '',
        active_simulation='active' if st.session_state.active_tab == 'simulation' else '',
        active_history='active' if st.session_state.active_tab == 'history' else '',
        active_chatbot='active' if st.session_state.active_tab == 'chatbot' else ''
    )
    st.markdown(header_html, unsafe_allow_html=True)

# Obtenir l'onglet actif (par défaut: prédiction)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "prediction"

# Utiliser les paramètres de requête pour la navigation
tab = st.query_params.get("tab", ["prediction"])[0] if hasattr(st, "query_params") else "prediction"
st.session_state.active_tab = tab

# Affichage de l'en-tête
render_header()

# ---------- Page 1 : Prédiction ----------
if st.session_state.active_tab == "prediction":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">📋</span><h2>Données du patient</h2></div>', unsafe_allow_html=True)
        
        # Boutons d'action rapide
        action_buttons_html = '''
        <div class="quick-action-buttons">
            <button class="btn-primary" id="analyze-btn" onclick="document.querySelector('#predict_btn').click()">🔍 Analyse rapide</button>
            <button class="btn-secondary" id="reset-btn">🔄 Réinitialiser</button>
            <button class="btn-secondary" id="example-btn">📊 Données d'exemple</button>
        </div>
        '''
        st.markdown(action_buttons_html, unsafe_allow_html=True)
        
        # Diviser les entrées en deux colonnes pour économiser de l'espace
        basic_cols, advanced_cols = st.columns(2)
        
        input_data = []
        field_labels = {
            "Pregnancies": "Grossesses",
            "Glucose": "Glucose (mg/dL)",
            "BloodPressure": "Pression artérielle (mmHg)",
            "SkinThickness": "Épaisseur de peau (mm)",
            "Insulin": "Insuline (μU/mL)",
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
            
            # Affichage du résultat
            result_html = f'''
            <div class="result-container result-{result_class}">
                <h3>Résultat de l'analyse</h3>
                <p class="result-text"><strong>{result_text}</strong> (Confiance: {probability:.2%})</p>
                
                <div class="risk-gauge-container">
                    <p><strong>Niveau de risque:</strong></p>
                    <div class="risk-gauge">
                        <div class="risk-gauge-fill" style="width: {probability*100}%;"></div>
                    </div>
                    <div class="risk-labels">
                        <span>Faible</span>
                        <span>Modéré</span>
                        <span>Élevé</span>
                    </div>
                </div>
            </div>
            '''
            st.markdown(result_html, unsafe_allow_html=True)
            
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
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
        
    with col2:
        # Carte d'information
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="icon">💡</span><h2>Saviez-vous?</h2></div>', unsafe_allow_html=True)
        
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
        
        # Application Mobile
        app_html = '''
        <div class="card" style="margin-top: 1.5rem;">
            <div class="section-header">
                <span class="icon">📱</span>
                <h2>Suivi de votre diabète</h2>
            </div>
            <p>Téléchargez notre application mobile DiabTrack pour suivre votre glycémie quotidiennement.</p>
            <a href="#" class="btn-primary" style="display: inline-block; text-decoration: none; text-align: center; margin-top: 0.5rem;">
                Télécharger l'application
            </a>
        </div>
        '''
        st.markdown(app_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture du main-container

# ---------- Page 2 : Explication du modèle ----------
elif st.session_state.active_tab == "explanation":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">📊</span><h2>Explication du Modèle Random Forest</h2></div>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    # Section supplémentaire expliquant les facteurs
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">📝</span><h2>Explication des facteurs principaux</h2></div>', unsafe_allow_html=True)
    
    factors = {
        "Glucose": "Mesure du taux de glucose dans le sang. Une valeur élevée (>140 mg/dL) peut indiquer un diabète.",
        "BMI": "Indice de Masse Corporelle. Un IMC > 30 indique une obésité, facteur de risque majeur.",
        "Age": "Le risque de diabète de type 2 augmente avec l'âge, particulièrement après 45 ans.",
        "Pregnancies": "Nombre de grossesses. Chaque grossesse augmente légèrement le risque de diabète.",
        "DiabetesPedigreeFunction": "Mesure du risque génétique basée sur les antécédents familiaux.",
        "BloodPressure": "Mesure de la pression artérielle. L'hypertension est souvent associée au diabète.",
        "Insulin": "Mesure de la réponse à l'insuline. Des niveaux élevés peuvent indiquer une résistance à l'insuline.",
        "SkinThickness": "L'épaisseur du pli cutané peut être un indicateur de la répartition des graisses corporelles."
    }
    
    cols = st.columns(3)
    for i, (factor, description) in enumerate(factors.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="factor-card">
                <h3>{factor}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    # Section graphique
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">📈</span><h2>Distribution des caractéristiques</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Sélectionnez une caractéristique à visualiser:",
            options=columns,
            index=1  # Glucose par défaut
        )
        
        # Histogramme de la distribution
        fig_data = pd.DataFrame({
            selected_feature: data[selected_feature],
            'Outcome': data['Outcome']
        })
        
        st.subheader(f"Distribution de {selected_feature}")
        st.bar_chart(fig_data.groupby('Outcome')[selected_feature].mean())
        st.caption("0 = Non diabétique, 1 = Diabétique")
    
    with col2:
        st.markdown("""
        ### 📊 Interprétation de la distribution
        
        Ce graphique montre comment la caractéristique sélectionnée varie entre les patients diabétiques (1) et non diabétiques (0).
        
        **Comment lire ce graphique:**
        - Une différence significative entre les deux groupes indique que cette caractéristique est fortement prédictive du diabète
        - Plus la différence est grande, plus le facteur est important dans le modèle
        
        **Exemple:**
        Pour le glucose, on observe généralement une valeur moyenne beaucoup plus élevée chez les patients diabétiques que chez les non-diabétiques.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture du main-container

# ---------- Page 3 : Simulation ----------
elif st.session_state.active_tab == "simulation":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">🔄</span><h2>Simulation de changements</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Simulez l'impact de changements dans certains facteurs de risque modifiables pour voir comment ils pourraient affecter votre risque de diabète.
        
        Commencez par saisir vos valeurs actuelles, puis utilisez les curseurs pour simuler des changements.
        """)
        
        base_input_data = []
        modified_input_data = []
        field_labels = {
            "Pregnancies": "Grossesses",
            "Glucose": "Glucose (mg/dL)",
            "BloodPressure": "Pression artérielle (mmHg)",
            "SkinThickness": "Épaisseur de peau (mm)",
            "Insulin": "Insuline (μU/mL)",
            "BMI": "IMC",
            "DiabetesPedigreeFunction": "Fonction pedigree diabétique",
            "Age": "Âge"
        }
        
        # Saisie des valeurs de base
        st.subheader("Valeurs actuelles")
        basic_cols, advanced_cols = st.columns(2)
        
        with basic_cols:
            for i, col in enumerate(columns[:4]):
                if col in field_labels:
                    default_val = data[col].median()
                    val = st.number_input(f"{field_labels[col]}", 
                                        value=float(default_val), 
                                        key=f"sim_base_{col}")
                    base_input_data.append(val)
                    modified_input_data.append(val)
        
        with advanced_cols:
            for i, col in enumerate(columns[4:8]):
                if col in field_labels:
                    default_val = data[col].median()
                    val = st.number_input(f"{field_labels[col]}", 
                                        value=float(default_val), 
                                        key=f"sim_base_{col}")
                    base_input_data.append(val)
                    modified_input_data.append(val)
        
        # Remplir les autres valeurs nécessaires (caractéristiques calculées)
        for i, col in enumerate(columns[8:]):
            if col == "Glucose_to_BMI":
                glucose_idx = list(columns).index("Glucose")
                bmi_idx = list(columns).index("BMI")
                val = base_input_data[glucose_idx] / max(base_input_data[bmi_idx], 0.1)
                base_input_data.append(val)
                modified_input_data.append(val)
            elif col == "Insulin_to_Glucose":
                insulin_idx = list(columns).index("Insulin")
                glucose_idx = list(columns).index("Glucose")
                val = base_input_data[insulin_idx] / max(base_input_data[glucose_idx], 0.1)
                base_input_data.append(val)
                modified_input_data.append(val)
            elif col == "Is_Elderly":
                age_idx = list(columns).index("Age")
                val = 1 if base_input_data[age_idx] > 50 else 0
                base_input_data.append(val)
                modified_input_data.append(val)
        
        # Calcul du risque de base
        base_array = np.array(base_input_data).reshape(1, -1)
        base_scaled = scaler.transform(base_array)
        base_probability = model.predict_proba(base_scaled)[0][1]
        
        # Simulation avec curseurs
        st.markdown("---")
        st.subheader("Simuler des changements")
        st.markdown("Modifiez ces facteurs pour voir l'impact sur votre risque:")
        
        modifiable_factors = ["Glucose", "BloodPressure", "BMI", "Insulin"]
        modifiable_labels = {
            "Glucose": "Réduire le glucose de",
            "BloodPressure": "Réduire la pression artérielle de",
            "BMI": "Réduire l'IMC de",
            "Insulin": "Réduire l'insuline de"
        }
        
        changes = {}
        glucose_change = 0
        bmi_change = 0
        insulin_change = 0
        
        for factor in modifiable_factors:
            if factor in columns:
                idx = list(columns).index(factor)
                max_val = base_input_data[idx] * 0.5  # Maximum 50% de réduction
                changes[factor] = st.slider(
                    modifiable_labels.get(factor, f"Réduire {factor} de"),
                    0.0, 
                    float(max_val),
                    0.0,
                    key=f"sim_change_{factor}"
                )
                
                # Mettre à jour les valeurs modifiées
                modified_input_data[idx] = base_input_data[idx] - changes[factor]
                
                # Stocker les changements pour recalculer les caractéristiques dérivées
                if factor == "Glucose":
                    glucose_change = changes[factor]
                elif factor == "BMI":
                    bmi_change = changes[factor]
                elif factor == "Insulin":
                    insulin_change = changes[factor]
        
        # Mettre à jour les caractéristiques dérivées
        for i, col in enumerate(columns[8:]):
            idx = list(columns).index(col)
            if col == "Glucose_to_BMI":
                glucose_idx = list(columns).index("Glucose")
                bmi_idx = list(columns).index("BMI")
                new_glucose = base_input_data[glucose_idx] - glucose_change
                new_bmi = base_input_data[bmi_idx] - bmi_change
                modified_input_data[idx] = new_glucose / max(new_bmi, 0.1)
            elif col == "Insulin_to_Glucose":
                insulin_idx = list(columns).index("Insulin")
                glucose_idx = list(columns).index("Glucose")
                new_insulin = base_input_data[insulin_idx] - insulin_change
                new_glucose = base_input_data[glucose_idx] - glucose_change
                modified_input_data[idx] = new_insulin / max(new_glucose, 0.1)
        
        # Calcul du risque modifié
        mod_array = np.array(modified_input_data).reshape(1, -1)
        mod_scaled = scaler.transform(mod_array)
        mod_probability = model.predict_proba(mod_scaled)[0][1]
        
        # Affichage des résultats
        risk_reduction = base_probability - mod_probability
        risk_reduction_percent = (risk_reduction / base_probability) * 100 if base_probability > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risque initial", f"{base_probability:.2%}")
        with col2:
            st.metric("Risque après changements", f"{mod_probability:.2%}", 
                    f"-{risk_reduction_percent:.1f}%", delta_color="inverse")
        
        # Visualisation de la comparaison
        st.markdown("### Comparaison des risques")
        chart_data = pd.DataFrame({
            'Scénario': ['Actuel', 'Après changements'],
            'Risque': [base_probability, mod_probability]
        })
        st.bar_chart(chart_data.set_index('Scénario'))
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h3>📋 Comment utiliser la simulation</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Saisissez vos valeurs actuelles** dans le formulaire de gauche
        2. **Utilisez les curseurs** pour simuler des réductions dans vos facteurs de risque
        3. **Observez l'impact** sur votre pourcentage de risque
        
        Cette simulation vous aide à comprendre quels changements pourraient avoir le plus grand impact sur votre risque de diabète.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<h3>💡 Bonnes pratiques pour réduire le risque</h3>', unsafe_allow_html=True)
        st.markdown("""
        **Pour réduire votre glycémie:**
        - Limitez les sucres ajoutés et les glucides raffinés
        - Augmentez votre activité physique (30 min/jour)
        - Maintenez un poids santé
        
        **Pour améliorer votre IMC:**
        - Adoptez une alimentation équilibrée
        - Pratiquez une activité physique régulière
        - Limitez la taille des portions
        
        **Pour réduire votre pression artérielle:**
        - Limitez votre consommation de sel
        - Évitez l'alcool et le tabac
        - Gérez votre stress
        
        **Pour améliorer votre sensibilité à l'insuline:**
        - Augmentez votre activité physique
        - Perdez du poids si nécessaire
        - Augmentez votre consommation de fibres
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture du main-container

# ---------- Page 4 : Historique ----------
elif st.session_state.active_tab == "history":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">🗂️</span><h2>Historique des analyses</h2></div>', unsafe_allow_html=True)
    
    # Initialiser l'historique s'il n'existe pas
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Afficher un message si l'historique est vide
    if not st.session_state.history:
        st.info("Aucune analyse n'a été effectuée pour le moment. Rendez-vous sur la page de prédiction pour analyser votre premier profil.")
    else:
        # Afficher l'historique
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"Analyse #{i+1} - {entry['date']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("#### Données du patient")
                    patient_data = pd.DataFrame({
                        'Facteur': [field_labels.get(col, col) for col in columns[:8]],
                        'Valeur': entry['data'][:8]
                    })
                    st.table(patient_data)
                
                with col2:
                    st.markdown("#### Résultat")
                    result_class = "positive" if entry['prediction'] == 1 else "negative"
                    result_text = "Risque détecté" if entry['prediction'] == 1 else "Pas de risque"
                    
                    result_html = f'''
                    <div class="result-card result-{result_class}">
                        <h3>{result_text}</h3>
                        <p class="result-prob">Probabilité: {entry['probability']:.2%}</p>
                    </div>
                    '''
                    st.markdown(result_html, unsafe_allow_html=True)
    
    # Bouton pour exporter l'historique
    if st.session_state.history:
        # Convertir l'historique en DataFrame
        history_df = pd.DataFrame([
            {**{field_labels.get(col, col): entry['data'][i] for i, col in enumerate(columns[:8])}, 
             'Probabilité': entry['probability'],
             'Prédiction': "Risque détecté" if entry['prediction'] == 1 else "Pas de risque",
             'Date': entry['date']}
            for entry in st.session_state.history
        ])
        
        # Fonction pour convertir DataFrame en CSV
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(history_df)
        
        st.download_button(
            label="📥 Télécharger l'historique (CSV)",
            data=csv,
            file_name="diabpredict_historique.csv",
            mime="text/csv",
        )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture du main-container

# ---------- Page 5 : ChatBot ----------
elif st.session_state.active_tab == "chatbot":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">💬</span><h2>Assistant virtuel DiabBot</h2></div>', unsafe_allow_html=True)
    
    # Initialiser l'historique de chat s'il n'existe pas
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Bonjour ! Je suis DiabBot, votre assistant virtuel pour répondre à vos questions sur le diabète. Comment puis-je vous aider aujourd'hui ?"}
        ]
    
    # Afficher l'historique de chat
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
    
    # Zone de saisie pour le chat
    user_input = st.text_input("Posez votre question ici...", key="user_query")
    
    # Dictionnaire de réponses prédéfinies
    responses = {
        "symptômes": "Les symptômes courants du diabète comprennent: soif excessive, mictions fréquentes, fatigue, vision floue, faim constante, perte de poids inexpliquée, cicatrisation lente des plaies, infections fréquentes et engourdissements dans les extrémités.",
        "types": "Il existe principalement trois types de diabète:\n\n1. **Diabète de type 1**: Maladie auto-immune où le pancréas produit peu ou pas d'insuline.\n\n2. **Diabète de type 2**: Le corps devient résistant à l'insuline ou ne produit pas assez d'insuline.\n\n3. **Diabète gestationnel**: Apparaît pendant la grossesse et disparaît généralement après l'accouchement.",
        "prévention": "Pour prévenir le diabète de type 2, vous pouvez:\n- Maintenir un poids santé\n- Faire de l'exercice régulièrement (au moins 30 minutes par jour)\n- Manger sainement (fruits, légumes, grains entiers)\n- Éviter les sucres ajoutés et les graisses saturées\n- Ne pas fumer\n- Limiter la consommation d'alcool",
        "alimentation": "Une alimentation adaptée pour les personnes diabétiques comprend:\n- Des légumes non féculents\n- Des fruits à faible indice glycémique\n- Des grains entiers\n- Des protéines maigres\n- Des graisses saines\n\nIl est important de limiter les aliments transformés, les sucres ajoutés et les glucides raffinés.",
        "traitement": "Le traitement du diabète peut inclure:\n- Des changements de mode de vie (alimentation, exercice)\n- Surveillance de la glycémie\n- Médicaments oraux\n- Insuline (injections ou pompe)\n- Éducation thérapeutique\n\nLe traitement spécifique dépend du type de diabète et de sa sévérité.",
        "complications": "Les complications potentielles du diabète comprennent:\n- Maladies cardiovasculaires\n- Néphropathie (maladie rénale)\n- Rétinopathie (problèmes de vision)\n- Neuropathie (lésions nerveuses)\n- Problèmes de pieds\n- Problèmes dentaires\n- Problèmes de peau\n- Démence\n\nUn bon contrôle de la glycémie peut réduire ces risques.",
        "urgence": "Consultez immédiatement un médecin si vous présentez des signes d'hyperglycémie sévère (glycémie très élevée) comme soif extrême, fatigue intense, vision floue, ou des signes d'hypoglycémie (glycémie très basse) comme confusion, tremblements, sueurs froides, étourdissements.",
        "sport": "L'activité physique est bénéfique pour les personnes diabétiques car elle:\n- Améliore la sensibilité à l'insuline\n- Aide à contrôler la glycémie\n- Réduit les risques cardiovasculaires\n- Aide à maintenir un poids santé\n\nVisez au moins 150 minutes d'activité modérée par semaine, après consultation avec votre médecin."
    }
    
    # Traitement de la requête utilisateur
    if user_input:
        # Ajouter le message utilisateur à l'historique
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Rechercher une réponse appropriée
        response_found = False
        for keyword, response in responses.items():
            if keyword in user_input.lower():
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                response_found = True
                break
        
        # Réponse par défaut si aucun mot-clé n'est trouvé
        if not response_found:
            default_response = "Je ne suis pas sûr de comprendre votre question. Pouvez-vous reformuler ou demander des informations sur les symptômes, types, prévention, alimentation, traitement, complications, urgences ou sport liés au diabète?"
            st.session_state.chat_history.append({"role": "assistant", "content": default_response})
        
        # Rafraîchir la page pour afficher la nouvelle réponse
        st.experimental_rerun()
    
    # Bouton pour réinitialiser la conversation
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Bonjour ! Je suis DiabBot, votre assistant virtuel pour répondre à vos questions sur le diabète. Comment puis-je vous aider aujourd'hui ?"}
        ]
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    # Section FAQ
    st.markdown('<div class="card fade-in" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="icon">❓</span><h2>Questions fréquentes</h2></div>', unsafe_allow_html=True)
    
    faq_items = [
        {"question": "Quels sont les principaux symptômes du diabète?", "keyword": "symptômes"},
        {"question": "Quels sont les différents types de diabète?", "keyword": "types"},
        {"question": "Comment peut-on prévenir le diabète?", "keyword": "prévention"},
        {"question": "Quelle alimentation est recommandée pour les diabétiques?", "keyword": "alimentation"},
        {"question": "Quand faut-il consulter en urgence?", "keyword": "urgence"}
    ]
    
    # Afficher les questions fréquentes sous forme de boutons
    cols = st.columns(3)
    for i, item in enumerate(faq_items):
        with cols[i % 3]:
            if st.button(item["question"]):
                st.session_state.chat_history.append({"role": "user", "content": item["question"]})
                st.session_state.chat_history.append({"role": "assistant", "content": responses[item["keyword"]]})
                st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture de la card
    
    st.markdown('</div>', unsafe_allow_html=True)  # Fermeture du main-container

# ---------- Ajouter un pied de page ----------
footer_html = '''
<footer class="footer">
    <div class="footer-content">
        <div class="footer-section">
            <h3>DiabPredict</h3>
            <p>© 2025 DiabPredict - Tous droits réservés</p>
        </div>
        <div class="footer-section">
            <h3>Contact</h3>
            <p>Email: contact@diabpredict.fr</p>
            <p>Tél: +33 1 23 45 67 89</p>
        </div>
        <div class="footer-section">
            <h3>Liens utiles</h3>
            <a href="#">À propos</a><br>
            <a href="#">Confidentialité</a><br>
            <a href="#">Conditions d'utilisation</a>
        </div>
    </div>
    <div class="footer-bottom">
        <p>Mentions légales: Cette application ne remplace pas un avis médical professionnel. Consultez toujours un médecin.</p>
    </div>
</footer>
'''
st.markdown(footer_html, unsafe_allow_html=True)

# JavaScript pour améliorer l'interactivité
js = '''
<script>
    // Fonctions pour les boutons d'action rapide
    document.getElementById('reset-btn').addEventListener('click', function() {
        // Réinitialiser tous les champs numériques
        const inputs = document.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            // Trouver le bouton de réinitialisation et cliquer dessus
            const resetButton = input.parentElement.querySelector('button');
            if (resetButton) resetButton.click();
        });
    });
    
    document.getElementById('example-btn').addEventListener('click', function() {
        // Exemples de valeurs pour un patient à risque
        const exampleData = {
            'pred_Pregnancies': 6,
            'pred_Glucose': 148,
            'pred_BloodPressure': 72,
            'pred_SkinThickness': 35,
            'pred_Insulin': 125,
            'pred_BMI': 33.6,
            'pred_DiabetesPedigreeFunction': 0.627,
            'pred_Age': 50,
            'pred_Glucose_to_BMI': 4.4,
            'pred_Insulin_to_Glucose': 0.84,
            'pred_Is_Elderly': 0
        };
        
        // Remplir les champs avec les valeurs d'exemple
        for (const [key, value] of Object.entries(exampleData)) {
            const input = document.querySelector(`[data-baseweb="input"][aria-labelledby="${key}"]`);
            if (input) {
                const inputField = input.querySelector('input');
                if (inputField) {
                    // Définir la valeur
                    inputField.value = value;
                    // Déclencher l'événement de changement
                    inputField.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }
    });
</script>
'''
st.markdown(js, unsafe_allow_html=True)
