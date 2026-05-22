import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Set up page configurations
st.set_page_config(
    page_title="EduTriage: AI-Powered Student Support Router",
    page_icon="🎓",
    layout="centered"
)

# 1. Load Data and Setup Encoders
@st.cache_resource
def load_and_setup_pipeline():
    csv_path = 'university_query.csv'
    model_path = 'model.pkl'
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        return None, None, None, None
        
    df = pd.read_csv(csv_path)
    model = joblib.load(model_path)
    
    # Initialize and fit encoders on text columns
    le_query = LabelEncoder().fit(df['Student_Query'])
    le_dept = LabelEncoder().fit(df['Department'])
    le_priority = LabelEncoder().fit(df['Priority_Label'])
    
    return model, le_query, le_dept, le_priority

model, le_query, le_dept, le_priority = load_and_setup_pipeline()

# UI Layout Header
st.title("🎓 EduTriage: AI-Powered Student Support Router")
st.write("Predict the priority label of incoming university student requests.")

# Error Fallback
if model is None:
    st.error("Missing Files! Ensure 'university_query.csv' and 'model.pkl' are in this folder.")
    st.stop()

st.markdown("---")
st.subheader("📥 Enter Query Parameters")

# 2. Add Inputs (Including the missing Days_To_Deadline column)
selected_query = st.selectbox("Select Student Query Type:", options=le_query.classes_)
selected_dept = st.selectbox("Select Target Department:", options=le_dept.classes_)

# Added the missing numeric slider input for deadlines
days_to_deadline = st.slider(
    "Days to Deadline:", 
    min_value=0, 
    max_value=30, 
    value=7, 
    step=1,
    help="Number of days remaining to resolve this query."
)

st.markdown("---")

# 3. Prediction Execution Block
if st.button("🔮 Predict Priority Level", type="primary"):
    try:
        # Transform categorical strings into numbers
        encoded_query = le_query.transform([selected_query])
        encoded_dept = le_dept.transform([selected_dept])
        
        # FIX: Rebuild DataFrame with all 3 required features in the correct order
        input_data = pd.DataFrame([{
            'Student_Query': int(encoded_query[0]),
            'Department': int(encoded_dept[0]),
            'Days_To_Deadline': float(days_to_deadline) # Missing column restored
        }])
        
        # Force column alignment order just to be absolutely safe
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(columns=model.feature_names_in_)
            
        # Run prediction
        raw_prediction = model.predict(input_data)
        
        # Convert numeric output back to text label
        decoded_priority = le_priority.inverse_transform(raw_prediction)[0]
        
        # Color coding visual blocks based on severity
        if "high" in str(decoded_priority).lower():
            st.error(f"### Predicted Priority: **{decoded_priority}**")
        elif "medium" in str(decoded_priority).lower():
            st.warning(f"### Predicted Priority: **{decoded_priority}**")
        else:
            st.success(f"### Predicted Priority: **{decoded_priority}**")
            
    except Exception as e:
        st.error(f"Prediction system failure. Details: {str(e)}")
