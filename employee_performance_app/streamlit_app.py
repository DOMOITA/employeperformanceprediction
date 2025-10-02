import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="INX Performance Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match your existing style
st.markdown("""
<style>
    /* Import Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Hero section styling */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .feature-icon {
        font-size: 3rem;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    /* Prediction form styling */
    .prediction-form {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Results styling */
    .prediction-result {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-title {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Get the directory of the current script for reliable file paths
BASE_DIR = Path(__file__).resolve().parent

# Load model and preprocessing objects
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load(BASE_DIR / 'models' / 'performance_model.pkl')
        scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
        feature_columns = joblib.load(BASE_DIR / 'models' / 'feature_columns.pkl')
        label_encoders = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
        return model, scaler, feature_columns, label_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, feature_columns, label_encoders = load_model_components()

# Field information dictionary (from your Flask app)
field_info = {
    'Age': {
        'placeholder': 'e.g., 28',
        'scale': 'Age in years (18-65)',
        'description': 'Employee age in years',
        'type': 'number',
        'min': 18, 'max': 65, 'default': 30
    },
    'Gender': {
        'placeholder': 'Select gender',
        'scale': 'Male, Female',
        'description': 'Employee gender',
        'type': 'select',
        'options': ['Male', 'Female']
    },
    'EducationBackground': {
        'placeholder': 'Select education field',
        'scale': 'Human Resources, Life Sciences, Marketing, Medical, Other, Technical Degree',
        'description': 'Field of education/specialization',
        'type': 'select',
        'options': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
    },
    'MaritalStatus': {
        'placeholder': 'Select marital status',
        'scale': 'Single, Married, Divorced',
        'description': 'Current marital status',
        'type': 'select',
        'options': ['Single', 'Married', 'Divorced']
    },
    'EmpDepartment': {
        'placeholder': 'Select department',
        'scale': 'Data Science, Development, Finance, Human Resources, Research & Development, Sales',
        'description': 'Department where employee works',
        'type': 'select',
        'options': ['Data Science', 'Development', 'Finance', 'Human Resources', 'Research & Development', 'Sales']
    },
    'EmpJobRole': {
        'placeholder': 'Select job role',
        'scale': 'Business Analyst, Data Scientist, Developer, Manager, Sales Executive, etc.',
        'description': 'Current job role/position',
        'type': 'select',
        'options': ['Business Analyst', 'Data Scientist', 'Delivery Manager', 'Developer', 'Finance Manager', 
                   'Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manager R&D', 
                   'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 
                   'Sales Representative', 'Senior Developer', 'Senior Manager R&D', 'Technical Architect', 'Technical Lead']
    },
    'BusinessTravelFrequency': {
        'placeholder': 'Select travel frequency',
        'scale': 'Travel_Rarely, Travel_Frequently, Non-Travel',
        'description': 'How often employee travels for business',
        'type': 'select',
        'options': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    },
    'DistanceFromHome': {
        'placeholder': 'e.g., 10',
        'scale': 'Distance in km (1-50)',
        'description': 'Distance from home to office in kilometers',
        'type': 'number',
        'min': 1, 'max': 50, 'default': 10
    },
    'EmpEducationLevel': {
        'placeholder': 'Select education level',
        'scale': 'Scale 1-5 (1=Below College, 5=PhD)',
        'description': 'Education level on numeric scale',
        'type': 'select',
        'options': ['1', '2', '3', '4', '5']
    },
    'EmpEnvironmentSatisfaction': {
        'placeholder': 'Select satisfaction level',
        'scale': 'Scale 1-4 (1=Low, 4=Very High)',
        'description': 'Satisfaction with work environment',
        'type': 'select',
        'options': ['1', '2', '3', '4']
    },
    'EmpHourlyRate': {
        'placeholder': 'e.g., 65',
        'scale': 'Rate per hour (30-100)',
        'description': 'Hourly rate in currency units',
        'type': 'number',
        'min': 30, 'max': 100, 'default': 65
    },
    'EmpJobInvolvement': {
        'placeholder': 'Select involvement level',
        'scale': 'Scale 1-4 (1=Low, 4=Very High)',
        'description': 'Level of job involvement',
        'type': 'select',
        'options': ['1', '2', '3', '4']
    },
    'EmpJobLevel': {
        'placeholder': 'Select job level',
        'scale': 'Level 1-5 (1=Entry, 5=Executive)',
        'description': 'Job level in organizational hierarchy',
        'type': 'select',
        'options': ['1', '2', '3', '4', '5']
    },
    'EmpJobSatisfaction': {
        'placeholder': 'Select satisfaction level',
        'scale': 'Scale 1-4 (1=Low, 4=Very High)',
        'description': 'Overall job satisfaction level',
        'type': 'select',
        'options': ['1', '2', '3', '4']
    },
    'NumCompaniesWorked': {
        'placeholder': 'e.g., 2',
        'scale': 'Number of companies (0-10)',
        'description': 'Total number of companies worked for',
        'type': 'number',
        'min': 0, 'max': 10, 'default': 2
    },
    'OverTime': {
        'placeholder': 'Select overtime status',
        'scale': 'Yes, No',
        'description': 'Whether employee works overtime',
        'type': 'select',
        'options': ['Yes', 'No']
    },
    'EmpLastSalaryHikePercent': {
        'placeholder': 'e.g., 15',
        'scale': 'Percentage (10-25%)',
        'description': 'Last salary hike percentage',
        'type': 'number',
        'min': 10, 'max': 25, 'default': 15
    },
    'EmpRelationshipSatisfaction': {
        'placeholder': 'Select satisfaction level',
        'scale': 'Scale 1-4 (1=Low, 4=Very High)',
        'description': 'Satisfaction with workplace relationships',
        'type': 'select',
        'options': ['1', '2', '3', '4']
    },
    'TotalWorkExperienceInYears': {
        'placeholder': 'e.g., 5',
        'scale': 'Years of experience (0-40)',
        'description': 'Total work experience in years',
        'type': 'number',
        'min': 0, 'max': 40, 'default': 5
    },
    'TrainingTimesLastYear': {
        'placeholder': 'e.g., 3',
        'scale': 'Number of trainings (0-10)',
        'description': 'Training sessions attended last year',
        'type': 'number',
        'min': 0, 'max': 10, 'default': 3
    },
    'EmpWorkLifeBalance': {
        'placeholder': 'Select balance level',
        'scale': 'Scale 1-4 (1=Bad, 4=Best)',
        'description': 'Work-life balance satisfaction',
        'type': 'select',
        'options': ['1', '2', '3', '4']
    },
    'ExperienceYearsAtThisCompany': {
        'placeholder': 'e.g., 3',
        'scale': 'Years at company (0-20)',
        'description': 'Years of experience at current company',
        'type': 'number',
        'min': 0, 'max': 20, 'default': 3
    },
    'ExperienceYearsInCurrentRole': {
        'placeholder': 'e.g., 2',
        'scale': 'Years in role (0-15)',
        'description': 'Years in current job role',
        'type': 'number',
        'min': 0, 'max': 15, 'default': 2
    },
    'YearsSinceLastPromotion': {
        'placeholder': 'e.g., 1',
        'scale': 'Years since promotion (0-10)',
        'description': 'Years since last promotion',
        'type': 'number',
        'min': 0, 'max': 10, 'default': 1
    },
    'YearsWithCurrManager': {
        'placeholder': 'e.g., 2',
        'scale': 'Years with manager (0-15)',
        'description': 'Years working with current manager',
        'type': 'number',
        'min': 0, 'max': 15, 'default': 2
    },
    'Attrition': {
        'placeholder': 'Select attrition status',
        'scale': 'Yes, No',
        'description': 'Whether employee has left the company',
        'type': 'select',
        'options': ['Yes', 'No']
    }
}

def preprocess_input(form_data):
    """Preprocess input data for prediction (adapted from Flask app)"""
    X = []
    
    for col in feature_columns:
        if col == 'PerformanceRating':
            continue  # Skip the target variable
        elif col == 'EmpNumber':
            X.append(1000)  # Default employee number
        else:
            val = form_data.get(col, '')
            if col in label_encoders:
                try:
                    val = label_encoders[col].transform([val])[0]
                except ValueError:
                    st.warning(f"Unseen label '{val}' for feature '{col}'. Using default value.")
                    val = 0
            else:
                val = float(val)
            X.append(val)
    
    X = np.array(X).reshape(1, -1)
    # Create DataFrame with feature names to avoid sklearn warning
    feature_names_for_prediction = [col for col in feature_columns if col not in ['PerformanceRating']]
    X_df = pd.DataFrame(X, columns=feature_names_for_prediction)
    X = scaler.transform(X_df)
    return X

# Sidebar navigation
st.sidebar.markdown("# 📊 INX Performance Hub")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ['🏠 Home', '🔮 Predict', '💡 Recommendations', 'ℹ️ About'])

# Home Page
if page == '🏠 Home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">
            <i class="fas fa-chart-line"></i>
            Welcome to INX Performance Hub
        </h1>
        <p class="hero-subtitle">
            Advanced Employee Performance Management & Predictive Analytics
        </p>
        <p>
            Empowering HR professionals with data-driven insights to enhance employee performance, 
            drive organizational excellence, and foster a culture of continuous improvement.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## Our HR Solutions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <i class="fas fa-chart-bar"></i>
            </div>
            <h3>Performance Analytics</h3>
            <p>Comprehensive analysis of employee performance across departments with actionable insights and trends.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h3>AI-Powered Predictions</h3>
            <p>Machine learning algorithms to predict employee performance and identify key improvement areas.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <i class="fas fa-lightbulb"></i>
            </div>
            <h3>Smart Recommendations</h3>
            <p>Personalized recommendations to improve employee engagement and performance outcomes.</p>
        </div>
        """, unsafe_allow_html=True)

# Predict Page
elif page == '🔮 Predict':
    st.markdown("# 🔮 Employee Performance Prediction")
    st.markdown("Enter employee details below to predict their performance rating.")
    
    if model is None:
        st.error("Model not loaded. Please check if the model files exist.")
    else:
        # Create prediction form
        with st.form("prediction_form"):
            st.markdown("### Employee Information")
            
            # Get input features (exclude target variable and EmpNumber)
            input_features = [col for col in feature_columns if col not in ['PerformanceRating', 'EmpNumber']]
            
            form_data = {}
            
            # Create input fields in columns
            cols = st.columns(2)
            for idx, feature in enumerate(input_features):
                col = cols[idx % 2]
                
                field_data = field_info.get(feature, {})
                field_type = field_data.get('type', 'text')
                description = field_data.get('description', feature)
                
                with col:
                    if field_type == 'select':
                        options = field_data.get('options', [])
                        form_data[feature] = st.selectbox(
                            f"{feature.replace('Emp', '').replace('_', ' ')}",
                            options,
                            help=description
                        )
                    elif field_type == 'number':
                        min_val = field_data.get('min', 0)
                        max_val = field_data.get('max', 100)
                        default_val = field_data.get('default', min_val)
                        form_data[feature] = st.number_input(
                            f"{feature.replace('Emp', '').replace('_', ' ')}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            help=description
                        )
            
            submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)
            
            if submitted:
                try:
                    # Preprocess input and make prediction
                    X = preprocess_input(form_data)
                    prediction = model.predict(X)[0]
                    
                    # Display prediction result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <div class="result-title">Predicted Performance Rating</div>
                        <div class="result-value">{prediction:.1f}</div>
                        <p>Based on the provided employee information</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Performance interpretation
                    if prediction >= 4:
                        st.success("🌟 Excellent Performance - This employee is likely to be a high performer!")
                    elif prediction >= 3:
                        st.info("👍 Good Performance - This employee shows solid performance capabilities.")
                    elif prediction >= 2:
                        st.warning("⚠️ Average Performance - Consider development opportunities.")
                    else:
                        st.error("📈 Below Average - May need additional support and training.")
                    
                    # Feature importance visualization (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("### Feature Importance")
                        feature_names = [col for col in feature_columns if col not in ['PerformanceRating', 'EmpNumber']]
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True).tail(10)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', 
                                   title='Top 10 Most Important Features for Prediction',
                                   orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

# Recommendations Page
elif page == '💡 Recommendations':
    st.markdown("# 💡 Performance Improvement Recommendations")
    
    st.markdown("""
    ## General Performance Enhancement Strategies
    
    Based on our analysis of employee performance data, here are key recommendations:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Individual Development
        - **Skill Assessment**: Regular evaluation of technical and soft skills
        - **Training Programs**: Targeted learning opportunities
        - **Mentorship**: Pairing with senior employees
        - **Goal Setting**: Clear, measurable objectives
        """)
        
        st.markdown("""
        ### 📊 Work Environment
        - **Work-Life Balance**: Flexible working arrangements
        - **Recognition Programs**: Acknowledge achievements
        - **Team Collaboration**: Foster positive relationships
        - **Feedback Culture**: Regular performance discussions
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Career Growth
        - **Promotion Pathways**: Clear advancement opportunities
        - **Cross-training**: Exposure to different departments
        - **Leadership Development**: Management skill building
        - **Project Ownership**: Increased responsibilities
        """)
        
        st.markdown("""
        ### 💼 Organizational Support
        - **Manager Training**: Effective leadership practices
        - **Resource Allocation**: Adequate tools and support
        - **Communication**: Open door policies
        - **Innovation**: Encourage creative thinking
        """)
    
    # Interactive recommendation generator
    st.markdown("## Personalized Recommendations")
    
    performance_level = st.selectbox(
        "Select Performance Level:",
        ["High Performer (4-5)", "Good Performer (3-4)", "Average Performer (2-3)", "Below Average (1-2)"]
    )
    
    department = st.selectbox(
        "Select Department:",
        ["Data Science", "Development", "Finance", "Human Resources", "Research & Development", "Sales"]
    )
    
    if st.button("Generate Recommendations"):
        recommendations = {
            "High Performer (4-5)": {
                "focus": "Leadership and Innovation",
                "actions": [
                    "Consider for leadership roles and special projects",
                    "Provide mentorship opportunities with junior staff",
                    "Offer advanced training and certifications",
                    "Include in strategic planning discussions"
                ]
            },
            "Good Performer (3-4)": {
                "focus": "Skill Enhancement and Growth",
                "actions": [
                    "Identify specific areas for improvement",
                    "Provide targeted training programs",
                    "Set challenging but achievable goals",
                    "Consider for cross-functional projects"
                ]
            },
            "Average Performer (2-3)": {
                "focus": "Support and Development",
                "actions": [
                    "Conduct detailed performance review",
                    "Provide additional training and support",
                    "Assign a mentor or coach",
                    "Create a performance improvement plan"
                ]
            },
            "Below Average (1-2)": {
                "focus": "Intensive Support and Monitoring",
                "actions": [
                    "Immediate intervention required",
                    "Comprehensive skills assessment",
                    "Intensive training and support",
                    "Regular monitoring and feedback"
                ]
            }
        }
        
        rec = recommendations.get(performance_level, {})
        
        st.success(f"**Focus Area**: {rec.get('focus', 'General Development')}")
        
        st.markdown("**Recommended Actions:**")
        for action in rec.get('actions', []):
            st.markdown(f"• {action}")

# About Page
elif page == 'ℹ️ About':
    st.markdown("# ℹ️ About INX Performance Hub")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Project Overview
        
        The INX Performance Hub is an advanced employee performance management system developed for 
        INX Future Inc. This application leverages machine learning algorithms to predict employee 
        performance ratings and provide actionable insights for HR professionals.
        
        ### Key Features
        - **Performance Prediction**: AI-powered predictions based on employee data
        - **Data Visualization**: Interactive charts and analytics
        - **Recommendations**: Personalized improvement strategies
        - **User-Friendly Interface**: Intuitive design for easy navigation
        
        ### Technology Stack
        - **Frontend**: Streamlit with custom CSS styling
        - **Backend**: Python with scikit-learn
        - **Data Processing**: Pandas and NumPy
        - **Visualization**: Plotly and Matplotlib
        
        ### Data Privacy
        All employee data is processed securely and in compliance with privacy regulations. 
        No personal information is stored permanently in the system.
        """)
    
    with col2:
        st.markdown("""
        ### Model Performance
        """)
        
        # Create a simple performance metrics display
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [0.85, 0.83, 0.86, 0.84]
        }
        
        fig = px.bar(
            metrics_data, 
            x='Metric', 
            y='Score',
            title='Model Performance Metrics',
            color='Score',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Contact Information
        For support or questions about this application, please contact:
        
        **Development Team**  
        INX Future Inc.  
        Email: support@inxfuture.com
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>© 2025 INX Future Inc. All rights reserved.</p>
        <p>Employee Performance Management System v1.0</p>
    </div>
    """, unsafe_allow_html=True)
