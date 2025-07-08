import streamlit as st
import pandas as pd
import joblib
# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Simple CSS for light colors
st.markdown("""
<style>
    .main-title {
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .prediction-price {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Housing.pkl')  # or Housing.joblib
        return model
    except FileNotFoundError:
        st.error("Model file 'Housing.pkl' not found.")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-title">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Input form
    st.subheader("Enter House Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=7000, step=100)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8], index=3)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=2)
        stories = st.selectbox("Stories", [1, 2, 3, 4], index=2)
        parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=2)
        furnishing = st.selectbox("Furnishing", ["unfurnished", "semi-furnished", "furnished"], index=2)
    
    with col2:
        mainroad = st.selectbox("Main Road Access", ["No", "Yes"], index=1)
        guestroom = st.selectbox("Guest Room", ["No", "Yes"], index=1)
        basement = st.selectbox("Basement", ["No", "Yes"], index=1)
        hotwaterheating = st.selectbox("Hot Water Heating", ["No", "Yes"], index=1)
        airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"], index=1)
        prefarea = st.selectbox("Preferred Area", ["No", "Yes"], index=1)
    
    # Predict button
    if st.button("Predict Price", use_container_width=True):
        # Convert inputs to numerical values
        mainroad_val = 1 if mainroad == "Yes" else 0
        guestroom_val = 1 if guestroom == "Yes" else 0
        basement_val = 1 if basement == "Yes" else 0
        hotwaterheating_val = 1 if hotwaterheating == "Yes" else 0
        airconditioning_val = 1 if airconditioning == "Yes" else 0
        prefarea_val = 1 if prefarea == "Yes" else 0
        
        furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
        furnishing_val = furnishing_map[furnishing]
        
        # Prepare input
        input_features = [
            area, bedrooms, bathrooms, stories, mainroad_val, 
            guestroom_val, basement_val, hotwaterheating_val, 
            airconditioning_val, parking, prefarea_val, furnishing_val
        ]
        
        # Make prediction
        try:
            prediction = model.predict([input_features])[0]
            
            # Display result
            st.markdown(f'''
            <div class="prediction-result">
                <h3>Predicted House Price</h3>
                <div class="prediction-price">${prediction:,.2f}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()