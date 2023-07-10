import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the linear regression model
model = joblib.load('linear_regression_model.pkl')

# Load the regression DataFrame
regression_df = pd.read_parquet('regression_df.parquet')

image = "mk-logo-black.png"
st.image(image, width=300)
# Add a centered subheader
st.subheader("Predictive Analytics for Manufacturing Optimization - IE University Impact Project")


# Add a caption
st.markdown("This tool has been developed to analyze and forecast the anticipated demand for different Mark'ennovy's products and associated parameters.")

# Create two columns for the dropdown boxes
col1, col2 = st.columns(2)

# Add a dropdown box
options = ['SAPHIR RX', 'EQUILIBRIA', 'QUATTRO 3M', 'XTENSA RX', 'SAPHIR 3M',
       'QUATTRO CONV', 'SPH 5', '5T', 'ET43', 'ES43', 'XTENSA',
       'GENTLE 80', 'GENTLE 59', 'METHAFILCON', 'HYDROGP', 'BLU:GEN',
       'BLU:KIDZ', 'BLU:SSENTIAL', 'MYLO', 'BRILLIANT', 'EDOF',
       'SEVEN RX']
selected_linprod = col1.selectbox("Name_LinProd", options)

options = ['SPHERIC', 'MULTIFOCAL TORIC', 'TORIC', 'MULTIFOCAL',
       'ASPHERIC']
selected_subfam = col2.selectbox("Name_SubFam", options)

# Add a dropdown box in the first column
options = ['2.2', '4.0', '5.0', '5.9', '6.0', '6.2', '6.3', '6.5', '6.6', '6.8', '7.0', '7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8', '7.9', '8.0', '8.1', '8.2', '8.3', '8.4', '8.5', '8.6', '8.7', '8.8', '8.9', '9.0', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.8', '10.0', '10.1', '16.3', '40.0']
selected_radius = col1.selectbox("Combined_Radius", options)

# Add a dropdown box in the second column
options_d = ["4.0", "6.0", "7.0", "8.0", "8.8", "9.0", "9.2", "9.4", "9.5", "9.6", "9.8", "10.0", "11.0", "11.4", "11.5", "12.0", "12.2", "12.5", "13.0", "13.1", "13.5", "13.8", "14.0", "14.2", "14.3", "14.4", "14.5", "14.7", "14.8", "15.0", "15.5", "16.0", "16.5", "17.0", "17.5", "18.0", "29.5", "29.8"]
selected_diameter = col2.selectbox("Combined_Diameter", options_d)

# Add a dropdown box in the first column
options_p = ["-35.0", "-34.0", "-33.5", "-33.0", "-30.5", "-30.0", "-29.75", "-29.5", "-29.25", "-29.0", "-28.75", "-28.5", "-28.25", "-28.0", "-27.75", "-27.5", "-27.25", "-27.0", "-26.75", "-26.5", "-26.25", "-26.0", "-25.75", "-25.5", "-25.25", "-25.0", "-24.75", "-24.5", "-24.25", "-24.0", "-23.75", "-23.5", "-23.25", "-23.0", "-22.75", "-22.5", "-22.25", "-22.0", "-21.75", "-21.5", "-21.25", "-21.0", "-20.75", "-20.5", "-20.25", "-20.0", "-19.75", "-19.5", "-19.25", "-19.0", "-18.75", "-18.5", "-18.25", "-18.0", "-17.75", "-17.5", "-17.0", "-16.75", "-16.5", "-16.25", "-16.0", "-15.75", "-15.5", "-15.25", "-15.0", "-14.75", "-14.5", "-14.25", "-14.0", "-13.75", "-13.5", "-13.25", "-13.0", "-12.75", "-12.5", "-12.25", "-12.0", "-11.75", "-11.5", "-11.25", "-11.0", "-10.75", "-10.5", "-10.25", "-10.0", "-9.75", "-9.5", "-9.25", "-9.0", "-8.75", "-8.5", "-8.25", "-8.0", "-7.75", "-7.5", "-7.25", "-7.0", "-6.75", "-6.5", "-6.25", "-6.0", "-5.75", "-5.5", "-5.25", "-5.0", "-4.75", "-4.62", "-4.5", "-4.38", "-4.25", "-4.15", "-4.0", "-3.91", "-3.75", "-3.67", "-3.5", "-3.44", "-3.25", "-3.2", "-3.0", "-2.97", "-2.75", "-2.73", "-2.6", "-2.5", "-2.25", "-2.15", "-2.1", "-2.02", "-2.0", "-1.95", "-1.79", "-1.75", "-1.55", "-1.5", "-1.25", "-1.0", "-0.75", "-0.61", "-0.5", "-0.25", "0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5", "2.75", "3.0", "3.25", "3.5", "3.75", "4.0", "4.25", "4.38", "4.5", "4.62", "4.75", "4.88", "5.0", "5.25", "5.5", "5.75", "6.0", "6.25", "6.5", "6.75", "7.0", "7.25", "7.5", "7.7", "7.75", "8.0", "8.25", "8.3", "8.5", "8.75", "9.0", "9.25", "9.5", "9.75", "10.0", "10.25", "10.5", "10.75", "11.0", "11.25", "11.5", "11.75", "12.0", "12.25", "12.5", "12.75", "13.0", "13.25", "13.5", "13.75", "14.0", "14.25", "14.35", "14.5", "14.75", "15.0", "15.25", "15.5", "15.75", "16.0", "16.25", "16.5", "16.75", "17.0", "17.25", "17.5", "17.75", "18.0", "18.25", "18.5", "18.75", "19.0", "19.25", "19.5", "19.63", "19.75", "20.0", "20.25", "20.5", "20.75", "21.0", "21.25", "21.5", "21.75", "22.0", "22.25", "22.5", "22.75", "23.0", "23.25", "23.5", "23.75", "24.0", "24.25", "24.5", "24.63", "24.75", "25.0", "25.25", "25.5", "25.75", "26.0", "26.13", "26.25", "26.5", "27.0", "27.25", "27.5", "27.75", "28.0", "28.25", "28.5", "28.75", "29.0", "29.25", "29.5", "29.75", "30.0", "31.5", "31.75", "32.0", "32.75", "33.0", "34.0", "35.0", "35.75", "36.0", "38.0"]
selected_power = col1.selectbox("Combined_Power", options_p)

# Add a dropdown box in the second column
options_c = ["-10.5", "-9.5", "-8.0", "-7.75", "-7.5", "-7.25", "-7.0", "-6.75", "-6.5", "-6.25", "-6.0", "-5.75", "-5.5", "-5.25", "-5.0", "-4.75", "-4.5", "-4.25", "-4.0", "-3.75", "-3.5", "-3.25", "-3.0", "-2.75", "-2.5", "-2.25", "-2.0", "-1.75", "-1.5", "-1.25", "-1.0", "-0.75", "-0.5", "-0.25", "0.0"]
selected_cylinder = col2.selectbox("Combined_Cylinder", options_c)

# Add a dropdown box in the second column
options_a = ["0.0", "1.0", "2.0", "3.0", "4.0", "4.0", "5.0", "6.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0", "25.0", "26.0", "27.0", "28.0", "29.0", "30.0", "31.0", "32.0", "33.0", "34.0", "35.0", "36.0", "37.0", "38.0", "39.0", "40.0", "41.0", "42.0", "43.0", "44.0", "45.0", "46.0", "47.0", "48.0", "49.0", "50.0", "51.0", "52.0", "53.0", "54.0", "55.0", "56.0", "57.0", "58.0", "59.0", "60.0", "61.0", "62.0", "63.0", "64.0", "65.0", "66.0", "67.0", "68.0", "69.0", "70.0", "71.0", "72.0", "73.0", "74.0", "75.0", "76.0", "77.0", "78.0", "79.0", "80.0", "81.0", "82.0", "83.0", "84.0", "85.0", "86.0", "87.0", "88.0", "89.0", "90.0", "91.0", "92.0", "93.0", "94.0", "95.0", "96.0", "97.0", "98.0", "99.0", "101.0", "102.0", "103.0", "104.0", "105.0", "106.0", "107.0", "108.0", "109.0", "110.0", "111.0", "112.0", "113.0", "114.0", "115.0", "116.0", "117.0", "118.0", "119.0", "120.0", "121.0", "122.0", "123.0", "124.0", "125.0", "126.0", "127.0", "128.0", "129.0", "130.0", "131.0", "132.0", "133.0", "134.0", "135.0", "136.0", "137.0", "138.0", "139.0", "140.0", "141.0", "142.0", "143.0", "144.0", "145.0", "146.0", "147.0", "148.0", "149.0", "150.0", "151.0", "152.0", "153.0", "154.0", "155.0", "156.0", "157.0", "158.0", "159.0", "161.0", "162.0", "163.0", "164.0", "165.0", "166.0", "167.0", "168.0", "169.0", "171.0", "172.0", "173.0", "174.0", "175.0", "176.0", "177.0", "178.0", "179.0"]
selected_axis = col1.selectbox("Combined_Axis", options_a)

# Add a dropdown box in the second column
options_m = ['1','2','3','4','5','6','7','8','9','10','11','12']
selected_month = col2.selectbox("Month", options_m)


# Concatenate selected options into a single string
selected_options = f"LinProd: {selected_linprod}, SubFam: {selected_subfam}, Radius: {selected_radius}, Diameter: {selected_diameter}, Power: {selected_power}, Cylinder: {selected_cylinder}, Axis: {selected_axis}, Month: {selected_month}"

# Display the selected options
st.write("Selected Options:", selected_options)

# Call the predict_qty function with the selected values
def predict_qty(linprod, subfam, radius, diameter, power, cylinder, axis, Month):
   # Fit and transform the categorical variables using one-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_linprod = encoder.fit_transform([[linprod]])
    encoded_subfam = encoder.transform([[subfam]])
    encoded_month = encoder.transform([[Month]])
    
    # Prepare the input DataFrame for prediction
    input_data = pd.DataFrame({
        'Name_LinProd': encoded_linprod.toarray().ravel(),
        'Name_SubFam': encoded_subfam.toarray().ravel(),
        'Combined_Radius': [radius],
        'Combined_Diameter': [diameter],
        'Combined_Power': [power],
        'Combined_Cylinder': [cylinder],
        'Combined_Axis': [axis],
        'Month': encoded_month.toarray().ravel()
    })

     # Perform the prediction
    predicted_qty = model.predict(input_data)
    return predicted_qty[0]

# Call the predict_qty function with the selected values
predicted_qty = predict_qty(selected_linprod, selected_subfam, selected_radius, selected_diameter, selected_power, selected_cylinder, selected_axis, selected_month)

# Round the predicted quantity
rounded_qty = round(predicted_qty)

# Create a button for prediction
if st.button('Predict The Demand'):
    # Call the predict_qty function with the selected values
    predicted_qty = predict_qty(selected_linprod, selected_subfam, selected_radius, selected_diameter, selected_power, selected_cylinder, selected_axis, selected_month)

    # Round the predicted quantity
    rounded_qty = round(predicted_qty)

    # Display the predicted SCHEDULED_QTY
    st.header(rounded_qty)