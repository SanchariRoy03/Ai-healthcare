from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import pandas as pd
import pickle


# flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")


svc = pickle.load(open('Models/svc.pkl','rb'))

# Load symptom severity data
df = pd.read_csv("datasets/Symptom-severity.csv")
valid_symptoms = set(df['Symptom'].str.strip().str.lower().str.replace(' ', '_').tolist())

all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'fatigue', 'weight_loss',
    'high_fever', 'lethargy', 'headache', 'red_spots', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
    'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
    'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremities', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
    'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite',
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking',
    'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]



symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#============================================================
# Synonym map
synonym_map = {
    "jaundice": "yellowing_of_eyes",
    "yellow skin": "yellowing_of_eyes",
    "tiredness": "fatigue",
    "loss of weight": "weight_loss",
    "weight gain": "weight_gain",
    "shiver": "shivering",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "nausea": "vomiting",
    "vomit": "vomiting",
    "stomach ache": "stomach_pain",
    "head pain": "headache",
    "cold": "continuous_sneezing",
    "sore throat": "throat_irritation",
    "diarrhea": "diarrhoea",
    "fever": "high_fever",
    "temperature": "high_fever",
    "eye pain": "pain_behind_the_eyes",
    "joint pain": "joint_pain",
    "pee burn": "burning_micturition",
    "weakness": "lethargy",
    "dizzy": "dizziness",
    "rash": "skin_rash",
    "chest ache": "chest_pain",
    "high fever": "high_fever",
    "sweat": "sweating",
    "sweats": "sweating",
    "blister": "red_spots",
    "red spots": "red_spots"
}
# ======================= Helper Functions ==========================

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    unknown_symptoms = []

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            unknown_symptoms.append(symptom)

    if unknown_symptoms:
        return None, unknown_symptoms

    prediction_index = svc.predict([input_vector])[0]
    return diseases_list[prediction_index], None

def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()
    return desc, pre, med, die, wrkout

# ======================= Routes ==========================

@app.route('/')
def index():
    return render_template('index.html', show_results=False)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        if not symptoms or symptoms.strip().lower() == "symptoms":
            return render_template("index.html", message="Please enter valid symptoms.")

        # ✅ Step 1: Normalize and clean input
        raw_symptoms = [s.strip().lower() for s in symptoms.split(',')]

        # ✅ Step 2: Map synonyms and replace spaces with underscores
        user_symptoms = []
        unknown_symptoms = []

        for s in raw_symptoms:
            s_key = s.strip().lower()
            s_key = synonym_map.get(s_key, s_key)  # Map known synonyms
            s_key = s_key.replace(" ", "_")        # Normalize to model format

            if s_key in symptoms_dict:
                user_symptoms.append(s_key)
            else:
                unknown_symptoms.append(s)

        # ✅ Handle unknowns
        if unknown_symptoms:
            message = f"Unknown symptom(s): {', '.join(unknown_symptoms)}"
            return render_template("index.html", message=message)

        # ✅ Predict disease
        predicted_disease, _ = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)
        my_precautions = [i for i in pre[0]]

        return render_template("index.html",
                               predicted_disease=predicted_disease,
                               dis_des=desc,
                               my_precautions=my_precautions,
                               medications=med,
                               my_diet=die,
                               workout=wrkout,
                               show_results=True)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email in users and users[email]['password'] == password:
            session['user'] = users[email]['name']
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password", "danger")

    return render_template("accounts.html")

users = {}

@app.route('/register', methods=['POST'])
def register():
    global users
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    if email in users:
        flash("Email already registered", "warning")
    else:
        users[email] = {'name': name, 'password': password}
        flash("Registration successful! Please login.", "success")

    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        print(f"Received message from {name} ({email}): {message}")

        flash("✅ Your message has been sent successfully!", "success")
        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)