import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

REQUIREMENTS_FILE = "requirements.txt"

# Set page configuration
st.set_page_config(page_title="Store and Product Classifier", layout="centered")

def store_classifier_app():
    # Model Training Section
    np.random.seed(42)
    num_samples = 1000
    data = {
        'store_name': [f'Store_{i}' for i in range(num_samples)],
        'historical_sales': np.random.lognormal(10, 0.4, num_samples).astype(int),
        'customer_rating': np.clip(np.random.normal(4.2, 0.3, num_samples), 3.0, 5.0),
        'location_type': np.random.choice(['Urban', 'Suburban', 'Rural'], num_samples, p=[0.3, 0.5, 0.2]),
        'area_income': np.random.randint(25000, 150000, num_samples),
        'floor_size': np.random.choice([500, 1000, 2000, 3000, 5000], num_samples),
        'class': np.random.choice(['A', 'B', 'C'], num_samples, p=[0.2, 0.3, 0.5])
    }
    df = pd.DataFrame(data)

    features = ['historical_sales', 'customer_rating', 'location_type', 'area_income', 'floor_size']
    target = 'class'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['historical_sales', 'customer_rating', 'area_income', 'floor_size']),
            ('cat', OneHotEncoder(), ['location_type'])
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced'))
    ])
    model.fit(df[features], df[target])

    # Streamlit UI
    st.title("Brand Store Classifier", anchor=False)
    st.markdown("Enter store details to classify and get recommendations.")

    with st.form(key="store_classifier_form"):
        historical_sales = st.slider(
            "Historical Sales (₹)", min_value=50000, max_value=20000000, step=10000, value=100000,
            help="Annual sales in Indian Rupees"
        )
        customer_rating = st.slider(
            "Customer Rating", min_value=3.0, max_value=5.0, step=0.1, value=4.0,
            help="Average customer rating (3.0 to 5.0)"
        )
        location_type = st.selectbox(
            "Location Type", options=['Urban', 'Suburban', 'Rural'], help="Type of store location"
        )
        area_income = st.number_input(
            "Average Area Income (₹)", min_value=100000, max_value=20000000, step=50000, value=100000,
            help="Average income of the area in Indian Rupees"
        )
        floor_size = st.selectbox(
            "Floor Size (sq.ft)", options=[500, 1000, 2000, 3000, 5000], help="Store floor size in square feet"
        )
        price_to_retailer = st.number_input(
            "Price to Retailer (₹)", min_value=100, max_value=100000, step=100, value=5000,
            help="Price at which products are sold to the retailer"
        )
        mrp = st.number_input(
            "MRP (₹)", min_value=100, max_value=100000, step=100, value=6000,
            help="Maximum Retail Price for the products"
        )
        submit_button = st.form_submit_button("Classify Store")

    recommendations = {
        'A': {
            'placement': "Premium eye-level shelves in main aisles",
            'products': "Luxury items, new product launches, large packs",
            'pricing': "Premium pricing recommended (Price to Retailer: ₹{price_to_retailer}, MRP: ₹{mrp})"
        },
        'B': {
            'placement': "Mid-store displays and end caps",
            'products': "Popular mainstream products, family packs",
            'pricing': "Competitive pricing strategy (Price to Retailer: ₹{price_to_retailer}, MRP: ₹{mrp})"
        },
        'C': {
            'placement': "Checkout area and value sections",
            'products': "Small packs, impulse buys, essential items",
            'pricing': "Value pricing with promotions (Price to Retailer: ₹{price_to_retailer}, MRP: ₹{mrp})"
        }
    }

    if submit_button:
        if price_to_retailer > mrp:
            st.error(f"Error: Price to Retailer (₹{price_to_retailer:.2f}) cannot be higher than MRP (₹{mrp:.2f}). Please adjust the values.")
        else:
            input_data = pd.DataFrame([{
                'historical_sales': historical_sales,
                'customer_rating': customer_rating,
                'location_type': location_type,
                'area_income': area_income,
                'floor_size': floor_size
            }])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            st.subheader("Prediction Results", anchor=False)
            st.write(f"🎯 **Predicted Store Class**: {prediction}")
            st.write("📊 **Classification Confidence**:")
            for cls, confidence in zip(model.classes_, proba):
                st.write(f"  • {cls}: {confidence:.1%}")
            st.write("📈 **Recommended Strategy**:")
            st.write(f"  • **Placement**: {recommendations[prediction]['placement']}")
            st.write(f"  • **Products**: {recommendations[prediction]['products']}")
            st.write(f"  • **Pricing**: {recommendations[prediction]['pricing'].format(price_to_retailer=price_to_retailer, mrp=mrp)}")
            st.write("💡 **Interpretation Guide**:")
            st.write("  • **Class A**: Premium Hypermarkets/Modern Trade")
            st.write("  • **Class B**: Semi-premium Large Stores")
            st.write("  • **Class C**: Small Kirana/General Stores")

    st.markdown("---")
    st.subheader("Instructions", anchor=False)
    st.write("""
    1. Use the sliders/inputs to describe your target store.
    2. Ensure Price to Retailer is not higher than MRP.
    3. Click 'Classify Store' to get instant recommendations.
    4. Repeat for different store scenarios.
    """)

def product_placement_advisor():
    # Model Training Section
    np.random.seed(42)
    num_samples = 2000
    product_data = {
        'product_size': np.random.choice(['Small', 'Medium', 'Large'], num_samples, p=[0.5, 0.3, 0.2]),
        'best_season': np.random.choice(['All-Season', 'Summer', 'Winter', 'Festive'], num_samples),
        'mrp': np.random.lognormal(6, 0.5, num_samples).astype(int),
        'retailer_price': np.random.lognormal(6, 0.5, num_samples).astype(int) * np.random.uniform(0.4, 0.7, num_samples),
        'online_sales_ratio': np.random.beta(2, 5, num_samples),
        'shelf_life_days': np.random.randint(7, 365, num_samples),
        'target_store_class': np.random.choice(['A', 'B', 'C'], num_samples, p=[0.3, 0.4, 0.3])
    }
    df = pd.DataFrame(product_data)
    df['retailer_price'] = df['mrp'] * np.random.uniform(0.4, 0.7, num_samples)

    features = ['product_size', 'best_season', 'mrp', 'retailer_price', 'online_sales_ratio', 'shelf_life_days']
    target = 'target_store_class'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['mrp', 'retailer_price', 'online_sales_ratio', 'shelf_life_days']),
            ('cat', OneHotEncoder(), ['product_size', 'best_season'])
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced'))
    ])
    model.fit(df[features], df[target])

    # Streamlit UI
    st.title("Product Store Matcher", anchor=False)
    st.markdown("Enter product details to find the best store classes for placement.")

    with st.form(key="product_placement_form"):
        product_size = st.selectbox(
            "Product Size", options=['Small', 'Medium', 'Large'], help="Size of the product"
        )
        best_season = st.selectbox(
            "Peak Season", options=['All-Season', 'Summer', 'Winter', 'Festive'], help="Best season for sales"
        )
        mrp = st.number_input(
            "MRP (₹)", min_value=50, max_value=5000, step=50, value=500, help="Maximum Retail Price"
        )
        retailer_price = st.number_input(
            "Retailer Price (₹)", min_value=20, max_value=3000, step=50, value=300,
            help="Price at which products are sold to the retailer"
        )
        online_sales_ratio = st.slider(
            "Online Sales Ratio", min_value=0.0, max_value=1.0, step=0.05, value=0.3,
            help="Ratio of sales from online channels"
        )
        shelf_life_days = st.slider(
            "Shelf Life (days)", min_value=7, max_value=365, step=7, value=90,
            help="Shelf life of the product in days"
        )
        submit_button = st.form_submit_button("Find Best Stores")

    store_profiles = {
        'A': {
            'type': 'Premium Modern Trade/Hypermarkets',
            'products': 'High-margin, branded, large packs >500 units',
            'margin': 'Needs >40% margin',
            'footfall': '10,000+ weekly customers'
        },
        'B': {
            'type': 'Semi-premium Large Stores',
            'products': 'Mid-range, 200-500 unit packs',
            'margin': '30-40% margin',
            'footfall': '5,000-10,000 weekly customers'
        },
        'C': {
            'type': 'Kirana/Small Stores',
            'products': 'Small packs <200 units, essential goods',
            'margin': '<30% margin',
            'footfall': '<5,000 weekly customers'
        }
    }

    if submit_button:
        if retailer_price > mrp:
            st.error(f"Error: Retailer Price (₹{retailer_price:.2f}) cannot be higher than MRP (₹{mrp:.2f}). Please adjust the values.")
        else:
            input_data = pd.DataFrame([{
                'product_size': product_size,
                'best_season': best_season,
                'mrp': mrp,
                'retailer_price': retailer_price,
                'online_sales_ratio': online_sales_ratio,
                'shelf_life_days': shelf_life_days
            }])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            st.subheader("Prediction Results", anchor=False)
            st.write(f"📦 **Optimal Store Class**: {prediction}")
            st.write("📊 **Suitability Scores**:")
            for cls, score in zip(model.classes_, proba):
                st.write(f"  • Class {cls}: {score:.1%}")
            st.write("🏬 **Store Class Profile**:")
            st.write(f"  • **Type**: {store_profiles[prediction]['type']}")
            st.write(f"  • **Ideal Products**: {store_profiles[prediction]['products']}")
            st.write(f"  • **Margin Requirements**: {store_profiles[prediction]['margin']}")
            st.write(f"  • **Typical Footfall**: {store_profiles[prediction]['footfall']}")
            st.write("💡 **Strategic Advice**:")
            if prediction == 'A':
                st.write("  • Focus on branding & in-store displays")
            elif prediction == 'B':
                st.write("  • Emphasize value propositions & promotions")
            else:
                st.write("  • Prioritize wide distribution & affordability")

    st.markdown("---")
    st.subheader("Instructions", anchor=False)
    st.write("""
    1. Describe your product using the inputs above.
    2. Ensure Retailer Price is not higher than MRP.
    3. Click 'Find Best Stores' to get instant placement recommendations.
    4. See strategic advice for each store class.
    """)

# Create tabbed interface
tab1, tab2 = st.tabs(["Store Classifier", "Product Placement Advisor"])

with tab1:
    store_classifier_app()

with tab2:
    product_placement_advisor()