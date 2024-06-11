
import streamlit as st
import pandas as pd
from joblib import load
 
# # Load the pre-trained pipeline
pipeline = load("C:/Users/Nitij/Downloads/trained_model.joblib")
 
def preprocess_data(data):
    # Replace 'na' values with 0
    data.replace('na', 0, inplace=True)
    data=data.apply(pd.to_numeric,errors='coerce')
    return data

def make_predictions_and_generate_report(data):
    # Preprocess the data
     data_preprocessed = preprocess_data(data)

    # Make predictions using the pre-trained pipeline
     predictions = pipeline.predict(data_preprocessed)

     suggestions = []
     for pred in predictions:
         if pred == 1:
             suggestions.append("Truck is fine.")
         else:
             suggestions.append("Truck needs service.")
 
     # Generate report
     report = pd.DataFrame({
         'Truck': range(1, len(suggestions) + 1),
         'Remarks': suggestions
     })
     return report
 
def main():
     st.title('Truck APS Repair Prediction Application')
 
     # Allow user to upload a CSV file
     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
 
     if uploaded_file is not None:
         data = pd.read_csv(uploaded_file)
 
         # Display uploaded data
         st.write('Uploaded Data')
         st.write(data)
 
         if st.button('Make Predictions and Generate Report'):
             # Make predictions and generate report
             report = make_predictions_and_generate_report(data)
 
             # Display the report
             st.write('Prediction Report')
             st.write(report)
 
if __name__ == "__main__":
    main()