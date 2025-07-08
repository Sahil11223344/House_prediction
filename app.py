import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

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
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">Get accurate house price predictions using advanced machine learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🔧 House Features</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Input features with better organization
    col1, col2 = st.columns(2)
    
    with st.sidebar:
        st.markdown("### 📏 **Basic Information**")
        area = st.number_input(
            "Area (sq ft)",
            min_value=1000,
            max_value=20000,
            value=7000,
            step=100,
            help="Total area of the house in square feet"
        )
        
        bedrooms = st.selectbox(
            "Number of Bedrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=3,
            help="Total number of bedrooms"
        )
        
        bathrooms = st.selectbox(
            "Number of Bathrooms",
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="Total number of bathrooms"
        )
        
        stories = st.selectbox(
            "Number of Stories",
            options=[1, 2, 3, 4],
            index=2,
            help="Number of floors/stories"
        )
        
        st.markdown("### 🏡 **Property Features**")
        mainroad = st.selectbox("Main Road Access", ["No", "Yes"], index=1)
        guestroom = st.selectbox("Guest Room", ["No", "Yes"], index=1)
        basement = st.selectbox("Basement", ["No", "Yes"], index=1)
        hotwaterheating = st.selectbox("Hot Water Heating", ["No", "Yes"], index=1)
        airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"], index=1)
        
        st.markdown("### 🚗 **Additional Features**")
        parking = st.selectbox(
            "Parking Spaces",
            options=[0, 1, 2, 3],
            index=2,
            help="Number of parking spaces"
        )
        
        prefarea = st.selectbox("Preferred Area", ["No", "Yes"], index=1)
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["unfurnished", "semi-furnished", "furnished"],
            index=2,
            help="Current furnishing status"
        )
    
    # Main content area
    with col1:
        st.markdown('<h2 class="sub-header">📊 Property Summary</h2>', unsafe_allow_html=True)
        
        # Create a summary display
        summary_data = {
            "Feature": ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking", "Furnishing"],
            "Value": [f"{area:,} sq ft", bedrooms, bathrooms, stories, parking, furnishingstatus.title()]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Feature checklist
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Property Features:**")
        features = [
            f"{'✅' if mainroad == 'Yes' else '❌'} Main Road Access",
            f"{'✅' if guestroom == 'Yes' else '❌'} Guest Room",
            f"{'✅' if basement == 'Yes' else '❌'} Basement",
            f"{'✅' if hotwaterheating == 'Yes' else '❌'} Hot Water Heating",
            f"{'✅' if airconditioning == 'Yes' else '❌'} Air Conditioning",
            f"{'✅' if prefarea == 'Yes' else '❌'} Preferred Area"
        ]
        for feature in features:
            st.markdown(f"- {feature}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Price Prediction</h2>', unsafe_allow_html=True)
        
        # Convert categorical inputs to numerical
        mainroad_val = 1 if mainroad == "Yes" else 0
        guestroom_val = 1 if guestroom == "Yes" else 0
        basement_val = 1 if basement == "Yes" else 0
        hotwaterheating_val = 1 if hotwaterheating == "Yes" else 0
        airconditioning_val = 1 if airconditioning == "Yes" else 0
        prefarea_val = 1 if prefarea == "Yes" else 0
        
        # Map furnishing status to numerical
        furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
        furnishing_val = furnishing_map[furnishingstatus]
        
        # Prepare input for prediction
        input_features = [
            area, bedrooms, bathrooms, stories, mainroad_val, 
            guestroom_val, basement_val, hotwaterheating_val, 
            airconditioning_val, parking, prefarea_val, furnishing_val
        ]
        
        if st.button("🔮 Predict House Price", use_container_width=True):
            try:
                # Make prediction
                prediction = model.predict([input_features])[0]
                
                # Display prediction with animation effect
                st.markdown(f'''
                <div class="prediction-box">
                    <h3>💰 Predicted House Price</h3>
                    <div class="prediction-value">${prediction:,.2f}</div>
                    <p>Based on the provided features and market analysis</p>
                </div>
                ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">Built with ❤️ using Streamlit • House Price Prediction ML Model</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()