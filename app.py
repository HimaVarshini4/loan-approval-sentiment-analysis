import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model_path = os.path.join('model/xgboostModel2.pkl')
    return joblib.load(model_path)

def ValuePredictor(data=pd.DataFrame):
    loaded_model = load_model()
    result = loaded_model.predict(data)
    return result[0]

@st.cache_data
def load_schema():
    schema_path = os.path.join(current_dir, 'columns_set.json')
    with open(schema_path, 'r') as f:
        cols = json.loads(f.read())
    return cols['data_columns']

def main():
    # Set page config
    st.set_page_config(
        page_title="Loan Approval Sentiment Analysis",
        page_icon="💰",
        layout="wide"
    )

    st.markdown("""
    <style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 0.5rem;
    }
    .stTextInput>div>input:focus, .stNumberInput>div>input:focus, .stSelectbox>div>select:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .stMarkdown {
        font-size: 18px;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Loan Approval Sentiment Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form(key='loan_form'):
            st.subheader("Personal Information")
            
            name = st.text_input("Your Name", placeholder="Enter your name")
            
            gender = st.selectbox("Gender", 
                                ["Male", "Female"],
                                index=None,
                                placeholder="Select gender")
            
            education = st.selectbox("Education",
                                   ["Graduate", "Not Graduate"],
                                   index=None,
                                   placeholder="Select education")
            
            self_employed = st.selectbox("Self Employed?",
                                       ["No", "Yes"],
                                       index=None,
                                       placeholder="Select employment status")
            
            marital_status = st.selectbox("Marital Status",
                                        ["Single", "Married"],
                                        index=None,
                                        placeholder="Select marital status")
            
            dependents = st.selectbox("Number of Dependents",
                                    ["0", "1", "2", "3+"],
                                    index=None,
                                    placeholder="Select number of dependents")
            
            property_area = st.selectbox("Property Area",
                                       ["Urban", "Semi-Urban", "Rural"],
                                       index=None,
                                       placeholder="Select property area")

            st.subheader("Financial Information")
            
            applicant_income = st.number_input("Applicant Income (USD/month)",
                                             min_value=0.0,
                                             format="%.2f")
            
            coapplicant_income = st.number_input("Co-applicant Income (USD/month)",
                                               min_value=0.0,
                                               format="%.2f")
            
            loan_amount = st.number_input("Loan Amount (USD)",
                                        min_value=0.0,
                                        format="%.2f")
            
            loan_term = st.number_input("Loan Term (days)",
                                      min_value=0,
                                      step=1)
            
            credit_history = st.selectbox("Credit History",
                                        ["All Debts Paid", "Not Paid"],
                                        index=None,
                                        placeholder="Select credit history")

            # Submit button
            submit_button = st.form_submit_button(label="Check Loan Status")

        if submit_button:
            schema_cols = load_schema().copy()

            if dependents:
                col = f'Dependents_{dependents}'
                if col in schema_cols:
                    schema_cols[col] = 1

            if property_area:
                col = f'Property_Area_{property_area}'
                if col in schema_cols:
                    schema_cols[col] = 1

            schema_cols['ApplicantIncome'] = applicant_income
            schema_cols['CoapplicantIncome'] = coapplicant_income
            schema_cols['LoanAmount'] = loan_amount
            schema_cols['Loan_Amount_Term'] = loan_term
            schema_cols['Gender_Male'] = 1 if gender == "Male" else 0
            schema_cols['Married_Yes'] = 1 if marital_status == "Married" else 0
            schema_cols['Education_Not Graduate'] = 1 if education == "Not Graduate" else 0
            schema_cols['Self_Employed_Yes'] = 1 if self_employed == "Yes" else 0
            schema_cols['Credit_History_1.0'] = 1 if credit_history == "All Debts Paid" else 0

            df = pd.DataFrame(
                data={k: [v] for k, v in schema_cols.items()},
                dtype=float
            )

            result = ValuePredictor(data=df)

            if int(result) == 1:
                st.success(f"Dear {name}, your loan is approved! 🎉")
            else:
                st.error(f"Sorry {name}, your loan is rejected. 😔")

    with col2:
        st.image("static/images/signin-image.jpg",
                caption="Loan Application",
                use_container_width=True)

if __name__ == '__main__':
    main()