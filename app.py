import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io
from sklearn.linear_model import LinearRegression

# Minimal CSS for styling
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
    color: #333;
}
h1, h2, h3 {
    color: #2c3e50;
}
.section {
    margin: 1em 0;
    padding: 1em;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #ffffff; /* Explicitly white background for sections */
}
.st-expander {
    background-color: #ffffff; /* White background for expanders */
    border: 1px solid #ddd;
    border-radius: 5px;
}
.stMetric {
    background-color: #ffffff !important; /* White background for metrics */
    padding: 1em;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #333 !important; /* Fallback for all text inside metrics */
}
.stMetric label {
    color: #333 !important; /* Dark text for metric labels */
}
.stMetric [data-testid="stMetricValue"] {
    color: #333 !important; /* Dark text for metric values */
}
.key-metrics-section {
    background-color: #ffffff !important; /* Ensure the entire Key Metrics section has a white background */
    padding: 1em;
    color: #333 !important; /* Fallback for all text in the section */
}
.key-metrics-section * {
    color: #333 !important; /* Force all text in the section to be dark */
}
.logout-button {
    position: fixed;
    bottom: 10px;
    width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for login and filtered data
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = None

# Login Page
if not st.session_state.logged_in:
    st.title("Health Claim Cost Prediction - Login")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Hardcoded credentials for simplicity (in a real app, use a secure database)
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()  # Rerun the app to show the main content
        else:
            st.error("Invalid username or password. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Load the dataset and model with error handling
    try:
        data = pd.read_csv("synthea_data.csv", dtype={"DIAGNOSIS1": str, "DIAGNOSIS2": str})
        model = joblib.load("ensemble_claim_cost_model.pkl")
    except Exception as e:
        st.error(f"Error loading dataset or model: {e}")
        st.stop()

    # Preprocess the data (aligned with the HTML file)
    try:
        # Define required columns
        required_columns = ["START", "STOP", "AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                            "ENCOUNTERCLASS", "CODE", "REASONCODE", "CODE_1", "DESCRIPTION", "PROVIDERID", 
                            "DIAGNOSIS1", "DIAGNOSIS2", "TOTALCOST"]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dataset is missing required columns: {missing_columns}")
            st.stop()

        # Handle patient ID column
        patient_id_column = None
        possible_patient_columns = ["PATIENTID", "PATIENT_ID", "Id", "ID", "PATIENT"]
        for col in possible_patient_columns:
            if col in data.columns:
                patient_id_column = col
                break
        
        if patient_id_column:
            data = data.rename(columns={patient_id_column: "PATIENT"})
        else:
            st.warning("No patient ID column found. Creating a placeholder PATIENT column.")
            data["PATIENT"] = [f"patient_{i}" for i in range(len(data))]
        
        required_columns = ["START", "STOP", "PATIENT", "AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                            "ENCOUNTERCLASS", "CODE", "REASONCODE", "CODE_1", "DESCRIPTION", "PROVIDERID", 
                            "DIAGNOSIS1", "DIAGNOSIS2", "TOTALCOST"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dataset is missing required columns: {missing_columns}")
            st.stop()

        # Calculate encounter duration
        data["ENCOUNTER_DURATION"] = (pd.to_datetime(data["STOP"]) - pd.to_datetime(data["START"])).dt.days

        # Convert DIAGNOSIS1 and DIAGNOSIS2 to strings
        data["DIAGNOSIS1"] = data["DIAGNOSIS1"].astype(str).fillna("None")
        data["DIAGNOSIS2"] = data["DIAGNOSIS2"].astype(str).fillna("None")

        # Check for chronic conditions (as per HTML file)
        data["CHRONIC_CONDITION"] = (data["DIAGNOSIS1"].str.contains("diabetes|hypertension", case=False, na=False).astype(int) | 
                                     data["DIAGNOSIS2"].str.contains("diabetes|hypertension", case=False, na=False).astype(int))
        data = data.fillna(0)

        # Extract year from START date for filtering and forecasting
        data["START_YEAR"] = pd.to_datetime(data["START"]).dt.year

        # Prepare features for one-hot encoding (as per HTML file)
        features = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", "ENCOUNTERCLASS", "CODE", "REASONCODE",
                    "CODE_1", "DESCRIPTION", "PROVIDERID", "DIAGNOSIS1", "DIAGNOSIS2", "ENCOUNTER_DURATION",
                    "CHRONIC_CONDITION"]
        X = data[features]
        X_encoded = pd.get_dummies(X)

        # Store the encoded feature names
        model_features = X_encoded.columns.tolist()
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        st.stop()

    # Initialize filtered data if not already set
    if st.session_state.filtered_data is None:
        st.session_state.filtered_data = data.copy()

    # Main App
    st.title("Health Claim Cost Prediction")

    # Add logout button in the sidebar at the bottom
    with st.sidebar:
        st.markdown("<div class='logout-button'>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Create tabs (8 tabs, Patient Details already removed)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Data Filters", "Key Metrics", "Claim Forecast", "Data Visualizations", "Resource Allocation", "Regional Claim Distributions", "Prediction Cost", "Data Export"])

    # Tab 1: Data Filters
    with tab1:
        st.header("Data Filters")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        years = list(range(1950, 2025))
        start_year = st.selectbox("Start Year:", years, index=1)
        end_year = st.selectbox("End Year:", years, index=len(years)-1)

        if st.button("Apply Year Range"):
            try:
                filtered_data = data[(data["START_YEAR"] >= start_year) & (data["START_YEAR"] <= end_year)]
                st.session_state.filtered_data = filtered_data
                st.write(f"Filtered Data: {len(filtered_data)} records")
            except Exception as e:
                st.error(f"Error applying year range filter: {e}")
        else:
            st.write(f"Filtered Data: {len(st.session_state.filtered_data)} records")

        # Debugging: Show unique races and encounter classes in the original and filtered data
        st.write("**Debugging Information:**")
        st.write(f"Unique Races in Original Data: {data['RACE'].unique()}")
        st.write(f"Unique Races in Filtered Data: {st.session_state.filtered_data['RACE'].unique()}")
        st.write(f"Unique Encounter Classes in Original Data: {data['ENCOUNTERCLASS'].unique()}")
        st.write(f"Unique Encounter Classes in Filtered Data: {st.session_state.filtered_data['ENCOUNTERCLASS'].unique()}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Key Metrics
    with tab2:
        st.header("Key Metrics")
        st.markdown("<div class='section key-metrics-section'>", unsafe_allow_html=True)

        filtered_data = st.session_state.filtered_data
        try:
            avg_claim_cost = filtered_data["TOTALCOST"].mean()
            total_claims = len(filtered_data)
            avg_age = filtered_data["AGE"].mean()
            total_patients = filtered_data["PATIENT"].nunique()
            avg_encounter_duration = filtered_data["ENCOUNTER_DURATION"].mean()
            most_common_diagnosis = filtered_data["DIAGNOSIS1"].mode()[0]

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Average Claim Cost", value=f"${avg_claim_cost:.2f}")
                st.metric(label="Total Number of Claims", value=total_claims)
            
            with col2:
                st.metric(label="Average Patient Age", value=f"{avg_age:.1f} years")
                st.metric(label="Total Number of Patients", value=total_patients)
            
            with col3:
                st.metric(label="Average Encounter Duration", value=f"{avg_encounter_duration:.1f} days")
                st.metric(label="Most Common Diagnosis", value=most_common_diagnosis)
        except Exception as e:
            st.error(f"Error calculating key metrics: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Claim Forecast
    with tab3:
        st.header("Claim Forecast")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            yearly_costs = data.groupby("START_YEAR")["TOTALCOST"].sum().reset_index()
            X = yearly_costs["START_YEAR"].values.reshape(-1, 1)
            y = yearly_costs["TOTALCOST"].values
            forecast_model = LinearRegression()
            forecast_model.fit(X, y)
            current_year = data["START_YEAR"].max()
            future_years = np.array([current_year + i for i in range(1, 6)]).reshape(-1, 1)
            forecasted_costs = forecast_model.predict(future_years)
            
            st.write("**Claim Cost Forecast for the Next 5 Years:**")
            forecast_df = pd.DataFrame({
                "Year": future_years.flatten(),
                "Forecasted Cost ($)": forecasted_costs
            })
            st.write(forecast_df)
            
            historical_df = pd.DataFrame({
                "Year": yearly_costs["START_YEAR"],
                "Cost": yearly_costs["TOTALCOST"],
                "Type": "Historical"
            })
            forecast_df_plot = pd.DataFrame({
                "Year": future_years.flatten(),
                "Cost": forecasted_costs,
                "Type": "Forecasted"
            })
            plot_df = pd.concat([historical_df, forecast_df_plot])
            st.line_chart(plot_df.set_index("Year")["Cost"])
        except Exception as e:
            st.error(f"Error generating claim forecast: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Data Visualizations
    with tab4:
        st.header("Data Visualizations")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.subheader("Total Cost Distribution")
        try:
            st.bar_chart(data["TOTALCOST"].value_counts())
        except Exception as e:
            st.error(f"Error creating Total Cost Distribution chart: {e}")

        st.subheader("Age vs Total Cost")
        try:
            st.scatter_chart(data[["AGE", "TOTALCOST"]])
        except Exception as e:
            st.error(f"Error creating Age vs Total Cost chart: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Resource Allocation (Updated)
    with tab5:
        st.header("Resource Allocation")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            # Debugging: Show unique encounter classes in the filtered data
            unique_encounter_classes = st.session_state.filtered_data["ENCOUNTERCLASS"].unique()
            st.write(f"**Unique Encounter Classes in Filtered Data:** {unique_encounter_classes}")

            # Group by ENCOUNTERCLASS
            resource_allocation = st.session_state.filtered_data.groupby("ENCOUNTERCLASS")["TOTALCOST"].sum().reset_index()
            st.write("**Total Claim Costs by Encounter Class:**")
            if resource_allocation.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(resource_allocation)
                st.bar_chart(resource_allocation.set_index("ENCOUNTERCLASS")["TOTALCOST"])
            
            avg_cost_by_encounter = st.session_state.filtered_data.groupby("ENCOUNTERCLASS")["TOTALCOST"].mean().reset_index()
            st.write("**Average Claim Cost by Encounter Class:**")
            if avg_cost_by_encounter.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(avg_cost_by_encounter)
        except Exception as e:
            st.error(f"Error analyzing resource allocation: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 6: Regional Claim Distributions (Updated)
    with tab6:
        st.header("Regional Claim Distributions")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        try:
            # Debugging: Show unique races in the filtered data
            unique_races = st.session_state.filtered_data["RACE"].unique()
            st.write(f"**Unique Races in Filtered Data:** {unique_races}")

            # Group by RACE
            regional_distribution = st.session_state.filtered_data.groupby("RACE")["TOTALCOST"].sum().reset_index()
            st.write("**Total Claim Costs by Race (Proxy for Region):**")
            if regional_distribution.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(regional_distribution)
                st.bar_chart(regional_distribution.set_index("RACE")["TOTALCOST"])
            
            avg_cost_by_race = st.session_state.filtered_data.groupby("RACE")["TOTALCOST"].mean().reset_index()
            st.write("**Average Claim Cost by Race (Proxy for Region):**")
            if avg_cost_by_race.empty:
                st.write("No data available for the selected year range. Please adjust the filters in the 'Data Filters' tab.")
            else:
                st.write(avg_cost_by_race)
        except Exception as e:
            st.error(f"Error analyzing regional claim distributions: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 7: Prediction Cost
    with tab7:
        st.header("Prediction Cost")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.subheader("Enter Patient Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 0, 100, 30, key="pred_age")
            gender = st.selectbox("Gender", ["M", "F"], key="pred_gender")
            race = st.selectbox("Race", data["RACE"].unique(), key="pred_race")
            ethnicity = st.selectbox("Ethnicity", data["ETHNICITY"].unique(), key="pred_ethnicity")
            income = st.slider("Income ($)", 0, 100000, 50000, key="pred_income")
            encounter_class = st.selectbox("Encounter Class", data["ENCOUNTERCLASS"].unique(), key="pred_encounter_class")
            code = st.selectbox("Procedure Code", data["CODE"].unique(), key="pred_code")
        
        with col2:
            reason_code = st.selectbox("Reason Code", data["REASONCODE"].unique(), key="pred_reason_code")
            code_1 = st.selectbox("Medication Code", data["CODE_1"].unique(), key="pred_code_1")
            description = st.selectbox("Description", data["DESCRIPTION"].unique(), key="pred_description")
            provider_id = st.selectbox("Provider ID", data["PROVIDERID"].unique(), key="pred_provider_id")
            diagnosis1 = st.selectbox("Diagnosis 1", data["DIAGNOSIS1"].unique(), key="pred_diagnosis1")
            diagnosis2 = st.selectbox("Diagnosis 2", data["DIAGNOSIS2"].unique(), key="pred_diagnosis2")
            encounter_duration = st.slider("Encounter Duration (days)", 0, 30, 1, key="pred_encounter_duration")

        chronic_condition = 1 if ("diabetes" in str(diagnosis1).lower() or "hypertension" in str(diagnosis1).lower() or
                                  "diabetes" in str(diagnosis2).lower() or "hypertension" in str(diagnosis2).lower()) else 0

        input_data = pd.DataFrame({
            "AGE": [age],
            "GENDER": [gender],
            "RACE": [race],
            "ETHNICITY": [ethnicity],
            "INCOME": [income],
            "ENCOUNTERCLASS": [encounter_class],
            "CODE": [code],
            "REASONCODE": [reason_code],
            "CODE_1": [code_1],
            "DESCRIPTION": [description],
            "PROVIDERID": [provider_id],
            "DIAGNOSIS1": [str(diagnosis1)],
            "DIAGNOSIS2": [str(diagnosis2)],
            "ENCOUNTER_DURATION": [encounter_duration],
            "CHRONIC_CONDITION": [chronic_condition]
        })

        try:
            categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "REASONCODE", 
                               "CODE_1", "DESCRIPTION", "PROVIDERID", "DIAGNOSIS1", "DIAGNOSIS2"]
            for col in categorical_cols:
                input_data[col] = input_data[col].astype(str)
            input_data_encoded = pd.get_dummies(input_data)
            input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)
        except Exception as e:
            st.error(f"Error encoding input data: {e}")
            st.stop()

        if st.button("Predict Cost"):
            try:
                prediction = model.predict(input_data_encoded)[0]
                st.write(f"### Predicted Claim Cost: ${prediction:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 8: Data Export
    with tab8:
        st.header("Data Export")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.write("Click the button below to download the filtered dataset as a CSV file.")
        try:
            buffer = io.BytesIO()
            st.session_state.filtered_data.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Export Data",
                data=buffer,
                file_name="filtered_data_export.csv",
                mime="text/csv",
                key="export_button",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error exporting data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)