
import joblib
import pandas as pd
import requests
import time
import streamlit as st


# Load the trained model
model = joblib.load("rf_model.pkl")  # Ensure this file is in the same directory

# Retrieve Fault Cause Mapping from the Model
fault_cause_mapping = model.fault_cause_mapping  # Now accessible in UI



def fetch_latest_data():
    url = "https://thingspeak.mathworks.com/channels/2815931/feed.json"
    try:
        response = requests.get(url)
        data = response.json()
        latest_entry = data['feeds'][-1]  # Get the most recent data entry
        return {
            "voltage": float(latest_entry["field1"]) if latest_entry["field1"] else 0.0,
            "current": float(latest_entry["field2"]) if latest_entry["field2"] else 0.0,
            "power": float(latest_entry["field3"]) if latest_entry["field3"] else 0.0,
            "energy": float(latest_entry["field4"]) if latest_entry["field4"] else 0.0,
            "frequency": float(latest_entry["field5"]) if latest_entry["field5"] else 50.0,
            "power_factor": float(latest_entry["field6"]) if latest_entry["field6"] else 1.0,
            "temperature": float(latest_entry["field7"]) if latest_entry["field7"] else 25.0
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch real-time data
latest_data = fetch_latest_data()


st.set_page_config(layout="wide")

# st.markdown(
#     """
#     <style>
#         body, .stApp {
#             background-color: #f9f9f9; /* Light Off-White Background */
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# st.markdown("""
#     <h1 style="text-align: left; font-size: 2.5em; margin-top: -60px;">Real-Time Transformer Monitoring & Fault Prediction</h1>
#     <hr style="border: 2px solid #ccc; margin-top: -10px;">
# """, unsafe_allow_html=True)

st.markdown("""
    <div style="
        background-color: black; 
        padding: 15px; 
        border-radius: 8px;
        text-align: left;
        margin-top: -60px;">
        <h1 style="
            font-size: 2.5em; 
            color: white; 
            margin: 0;">
            Real-Time Transformer Monitoring & Fault Prediction
        </h1>
    </div>
    
""", unsafe_allow_html=True)
#<hr style="border: 2px solid #ccc; margin-top: -10px;">



# Embedding ThingSpeak Graphs
st.markdown('<h4 style="margin-bottom: 5px;">Real-Time Sensor Data</h4>', unsafe_allow_html=True)



iframe_links = {
    "Voltage": "https://thingspeak.com/channels/2815931/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
    "Current": "https://thingspeak.com/channels/2815931/charts/2?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
    "Power": "https://thingspeak.com/channels/2815931/charts/3?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
#     "Energy": "https://thingspeak.com/channels/2815931/charts/4?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
    "Frequency": "https://thingspeak.com/channels/2815931/charts/5?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
    "Power Factor": "https://thingspeak.com/channels/2815931/charts/6?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line&update=15",
    "Temperature": "https://thingspeak.com/channels/2815931/charts/7?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&type=line"
}

col1, col2, col3 = st.columns([1, 1, 1])

# Row 1: Voltage Graph & Voltage Widget
# with col1:
#     st.markdown('<h5 style="margin-bottom: 5px;">Voltage</h5>', unsafe_allow_html=True)
#     st.components.v1.iframe(iframe_links["Voltage"], width=450, height=200)
# with col2:
#     st.markdown('<h5 style="margin-bottom: 5px;">Temperature</h5>', unsafe_allow_html=True)
#     st.components.v1.iframe(iframe_links["Temperature"], width=450, height=200)

with col1:
    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Voltage</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Voltage"]), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Temperature</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Temperature"]), unsafe_allow_html=True)







# Row 2: Temperature Graph & Temperature Widget
with col1:
    st.markdown('<h5 style="margin-bottom: 5px;">Voltage</h5>', unsafe_allow_html=True)
    st.components.v1.iframe("https://thingspeak.com/channels/2815931/widgets/1032359", width=200, height=200)
with col2:
    st.markdown('<h5 style="margin-bottom: 5px;">Temperature</h5>', unsafe_allow_html=True)
    st.components.v1.iframe("https://thingspeak.com/channels/2815931/widgets/1023341", width=200, height=200)

# Remaining Graphs Below
with col1:
    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Current</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Current"]), unsafe_allow_html=True)

    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Power</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Power"]), unsafe_allow_html=True)
#     st.subheader("Energy")
#     st.components.v1.iframe(iframe_links["Energy"], width=450, height=200)

with col2:
    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Frequency</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Frequency"]), unsafe_allow_html=True)

    st.markdown("""
        <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; overflow: hidden;">
            <h5 style="margin-bottom: 5px;">Power Factor</h5>
            <iframe src="{0}" width="450" height="200" style="border: none; overflow: hidden;" scrolling="no"></iframe>
        </div>
    """.format(iframe_links["Power Factor"]), unsafe_allow_html=True)

# Fault Prediction in the 3rd Column
with col3:
    # Fault Prediction Title
    st.markdown("""
        <div style="
            border: 2px solid #ccc; 
            padding: 15px; 
            border-radius: 8px; 
            background-color: black;
            text-align: center;">
            <h4 style="margin-bottom: 5px; color: white ">Fault Prediction</h4>
    """, unsafe_allow_html=True)
 
    # st.markdown('<h4 style="margin-bottom: 5px;">Fault Prediction</h4>', unsafe_allow_html=True)


    
    #prediction above form
        
    if latest_data:
        voltage = latest_data["voltage"]
        current = latest_data["current"]
        power = latest_data["power"]
        energy = latest_data["energy"]
        frequency = latest_data["frequency"]
        power_factor = latest_data["power_factor"]
        temp = latest_data["temperature"]
    else:
        st.warning("Using default values due to an error fetching data.")
        voltage, current, power, energy, frequency, power_factor, temp = 0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 25.0

    # Pass the fetched values to the ML model for prediction
    input_data = pd.DataFrame([[voltage, current, power, energy, frequency, power_factor, temp]],
                              columns=['voltage', 'current', 'power', 'energy', 'frequency', 'power factor', 'temp'])

    # Get the prediction automatically
    predicted_fault = model.predict(input_data)[0]
    predicted_cause = fault_cause_mapping.get(predicted_fault, "Unknown Cause")

#     # Display prediction results (Above the form)
#     st.write("##### **Predicted Fault:**", f":red[{predicted_fault}]")
#     st.write("##### **Cause:**", f":blue[{predicted_cause}]")

    st.markdown(f"""
    <div style="border: 2px solid #ccc; padding: 5px; border-radius: 8px; background-color: #F4F4FD; margin: 2px 0;">
        <h6 style="color: #999999; margin-bottom: 2px;">Predicted Fault:</h6>
        <p style="font-size: 18px; font-weight: bold; color: #666666; margin-bottom: 2px;">{predicted_fault}</p>
        <h6 style="color: #999999; margin-bottom: 2px;">Cause:</h6>
        <p style="font-size: 18px; font-weight: bold; color: #666666; margin-bottom: 2px;">{predicted_cause}</p>
    </div>
""", unsafe_allow_html=True)

    placeholder = st.empty()  # Create a placeholder for the form

    while True:
        with placeholder.form("fault_form"):
            st.text_input("Voltage (V)", value=voltage, disabled=True)
            st.text_input("Temperature (°C)", value=temp, disabled=True)
            st.text_input("Current (A)", value=current, disabled=True)
            st.text_input("Power (W)", value=power, disabled=True)
            st.text_input("Energy (kWh)", value=energy, disabled=True)
            st.text_input("Frequency (Hz)", value=frequency, disabled=True)
            st.text_input("Power Factor", value=power_factor, disabled=True)

            # Submit button (Runs automatically)
            st.form_submit_button("Refresh Data")

        time.sleep(30)  # Wait for 30 seconds
        st.rerun()  # Refresh the Streamlit app
    st.markdown("</div>", unsafe_allow_html=True)  # Close the styled div

    
    
