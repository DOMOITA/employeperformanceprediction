from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# Get the directory of the current script for reliable file paths
BASE_DIR = Path(__file__).resolve().parent

# Load model and preprocessing objects with absolute paths
try:
    model = joblib.load(BASE_DIR / 'models' / 'performance_model.pkl')
    scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    feature_columns = joblib.load(BASE_DIR / 'models' / 'feature_columns.pkl')
    label_encoders = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # You might want to handle this more gracefully in production

# Input field recommendations and scales - Only features used in model training
field_info = {
    'EmpNumber': {
        'placeholder': 'e.g., 1001',
        'scale': 'Employee ID (1000-9999)',
        'description': 'Unique employee identification number',
        'type': 'number'
    },
    'Age': {
        'placeholder': 'e.g., 28',
        'scale': 'Age in years (18-65)',
        'description': 'Employee age in years',
        'type': 'number'
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
        'options': ['Business Analyst', 'Data Scientist', 'Delivery Manager', 'Developer', 'Finance Manager', 'Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manager R&D', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative', 'Senior Developer', 'Senior Manager R&D', 'Technical Architect', 'Technical Lead']
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
        'type': 'number'
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
        'type': 'number'
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
        'type': 'number'
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
        'type': 'number'
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
        'type': 'number'
    },
    'TrainingTimesLastYear': {
        'placeholder': 'e.g., 3',
        'scale': 'Number of trainings (0-10)',
        'description': 'Training sessions attended last year',
        'type': 'number'
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
        'type': 'number'
    },
    'ExperienceYearsInCurrentRole': {
        'placeholder': 'e.g., 2',
        'scale': 'Years in role (0-15)',
        'description': 'Years in current job role',
        'type': 'number'
    },
    'YearsSinceLastPromotion': {
        'placeholder': 'e.g., 1',
        'scale': 'Years since promotion (0-10)',
        'description': 'Years since last promotion',
        'type': 'number'
    },
    'YearsWithCurrManager': {
        'placeholder': 'e.g., 2',
        'scale': 'Years with manager (0-15)',
        'description': 'Years working with current manager',
        'type': 'number'
    },
    'Attrition': {
        'placeholder': 'Select attrition status',
        'scale': 'Yes, No',
        'description': 'Whether employee has left the company',
        'type': 'select',
        'options': ['Yes', 'No']
    }
}

def get_placeholder(field_name):
    return field_info.get(field_name, {}).get('placeholder', f'Enter {field_name.lower()}')

def get_scale(field_name):
    return field_info.get(field_name, {}).get('scale', 'Enter appropriate value')

def get_description(field_name):
    return field_info.get(field_name, {}).get('description', f'Enter {field_name.replace("_", " ").lower()}')

def get_field_type(field_name):
    return field_info.get(field_name, {}).get('type', 'text')

def get_field_options(field_name):
    return field_info.get(field_name, {}).get('options', [])

def preprocess_input(form_data):
	X = []
	# Filter out only the target variable (PerformanceRating) but keep EmpNumber for model compatibility
	input_features_from_form = [col for col in feature_columns if col != 'PerformanceRating' and col != 'EmpNumber']
	
	for col in feature_columns:
		if col == 'PerformanceRating':
			continue  # Skip only the target variable
		elif col == 'EmpNumber':
			# Provide a default employee number since it's not in the form
			X.append(1000)  # Default employee number
		else:
			val = form_data.get(col, '')
			if col in label_encoders:
				try:
					val = label_encoders[col].transform([val])[0]
				except ValueError:
					# Handle unseen labels by using the most frequent class or 0
					print(f"Warning: Unseen label '{val}' for feature '{col}'. Using default value.")
					# Use the first class as default (usually the most frequent during training)
					val = 0
			else:
				val = float(val)
			X.append(val)
	print(f"Debug: Created {len(X)} features for prediction")
	X = np.array(X).reshape(1, -1)
	X = scaler.transform(X)
	return X

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	prediction = None
	if request.method == 'POST':
		X = preprocess_input(request.form)
		pred = model.predict(X)[0]
		prediction = f"Predicted Performance Rating: {pred}"
	
	# Filter out only the target variable (PerformanceRating) and EmpNumber from the form
	input_features = [col for col in feature_columns if col not in ['PerformanceRating', 'EmpNumber']]
	
	return render_template('predict.html', 
						   feature_columns=input_features, 
						   prediction=prediction,
						   get_placeholder=get_placeholder,
						   get_scale=get_scale,
						   get_description=get_description,
						   get_field_type=get_field_type,
						   get_field_options=get_field_options)

@app.route('/recommendations')
def recommendations():
	return render_template('recommendations.html')

if __name__ == '__main__':
	# For local development
	app.run(debug=False, host='0.0.0.0', port=5000)
else:
	# For production deployment (PythonAnywhere)
	application = app
