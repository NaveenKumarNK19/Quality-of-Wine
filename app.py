import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load the trained model
model = joblib.load('wine_quality_model.pkl')  # Ensure the model file is present

# Load the dataset
data = pd.read_csv(r"E:\STAT BRIO\Python\STUDENT\COMPLETED\Quality of Wine\Dataset\WineQT.csv")  # Replace with your actual dataset file name

# Set up the page configurations
st.set_page_config(
    page_title="Wine Quality Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the Home Page with additional insights and fun facts
def home_page():
    st.title("🍷 Wine Quality Prediction")
    st.header("Welcome to the Wine Quality Analysis Platform!")
    
    # Introduction
    st.markdown("""
        **This platform uses machine learning to predict the quality of wine based on its characteristics.**
        Whether you're a wine enthusiast, a connoisseur, or just someone curious about the science behind wine quality, you're in the right place!
    """)

    # Wine Insights
    st.subheader("🍇 What Makes a Great Wine?")
    st.markdown("""
    The quality of wine depends on various factors, including:
    - **Acidity**: A balanced acidity gives wine its crisp and refreshing taste.
    - **Sugar**: Residual sugar adds sweetness and affects body and texture.
    - **Tannins**: Contribute to the astringency and structure of red wines.
    - **Alcohol**: Influences the wine's body and warmth.
    - **pH Levels**: Affects the overall taste, with lower pH indicating sharper acidity.

    Fun fact: **Did you know that wine has been around for over 8,000 years?** The oldest known winery was found in Armenia, dating back to 4100 BCE!
    """)

    # Fun Facts
    st.subheader("🍷 Wine Trivia")
    st.markdown("""
    - **The world’s largest wine bottle** holds **1,590 liters** of wine. That’s equivalent to over 2,100 standard bottles!
    - The term “vintage” doesn’t mean “old.” It simply refers to the year the grapes were harvested.
    - **Red wine** gets its color from grape skins, while **white wine** is made by fermenting the juice without the skins.
    - A bottle of wine contains about **600-800 grapes**!

    🥂 **Cheers to learning something new while sipping your favorite vino!**
    """)

    # How the Prediction Works
    st.subheader("🔬 How Does This Platform Work?")
    st.markdown("""
    This platform uses a trained machine learning model to predict wine quality. Here's the magic in a nutshell:
    1. **Data Collection**: Thousands of wine samples were analyzed to identify key factors influencing quality.
    2. **Training**: A machine learning algorithm was trained using this data to recognize patterns.
    3. **Prediction**: Based on your input, the model predicts a quality score ranging from **1 to 10**, where higher scores mean better quality.

    ### Why Trust the Model?
    - It’s trained on a well-curated dataset.
    - High accuracy in predicting quality scores.

    🍷 Ready to see the science in action? Head to the "Prediction" tab and give it a try!
    """)

    # Humor Section
    st.subheader("😂 Wine Jokes (Because Why Not?)")
    st.markdown("""
    - **Why did the grape stop in the middle of the road?** It ran out of juice.
    - **What do you call a wine hangover?** The grape depression.
    - **Why don’t we ever tell secrets at a vineyard?** Too many grapes can hear through the grapevine.

    **Life’s too short for bad wine and no laughs!**
    """)

    # Call-to-action
    st.markdown("""
    ---
    🔮 **Explore more:** Use the navigation on the left to dive into predictions and wine science.
    🥂 *Enjoy your time on the platform and may your wine always be divine!* 🍇
    """)

# Define the Prediction Page
def prediction_page():
    st.title("🍷 Predict Wine Quality")
    st.markdown("### Enter the details of the wine below:")

    # Create two columns
    col1, col2 = st.columns(2)

    # Input fields for wine characteristics
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", min_value=4.6, max_value=15.9, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", min_value=0.12, max_value=1.58, step=0.01)
        citric_acid = st.number_input("Citric Acid (g/dm³)", min_value=0.0, max_value=1.0, step=0.01)
        residual_sugar = st.number_input("Residual Sugar (g/dm³)", min_value=0.9, max_value=15.5, step=0.1)
        chlorides = st.number_input("Chlorides (g/dm³)", min_value=0.012, max_value=0.611, step=0.001)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/dm³)", min_value=1.0, max_value=68.0, step=1.0)

    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/dm³)", min_value=6.0, max_value=289.0, step=1.0)
        density = st.number_input("Density (g/cm³)", min_value=0.99007, max_value=1.00369, step=0.00001, format="%.5f")
        ph = st.number_input("pH", min_value=2.74, max_value=4.01, step=0.01)
        sulphates = st.number_input("Sulphates (g/dm³)", min_value=0.33, max_value=2.0, step=0.01)
        alcohol = st.number_input("Alcohol (% vol)", min_value=8.4, max_value=14.9, step=0.1)

    # Button to predict quality
    if st.button("Predict Quality"):
        features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                             free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"Predicted Quality: {prediction} ⭐")

# Define the Analysis Page
def analysis_page():
    st.title("📊 Interactive Data Analysis: Wine Quality Insights")
    st.markdown("""
        Welcome to the interactive analysis section. Explore the dataset and uncover patterns that influence wine quality!
    """)

    # Show Dataset Overview
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.markdown(f"**Dataset Dimensions:** {data.shape[0]} rows and {data.shape[1]} columns")

    # Quality Distribution
    st.subheader("Wine Quality Distribution")
    fig = px.bar(
        data['quality'].value_counts().sort_index(),
        x=data['quality'].value_counts().sort_index().index,
        y=data['quality'].value_counts().sort_index().values,
        labels={'x': 'Quality', 'y': 'Count'},
        title="Wine Quality Counts",
        color=data['quality'].value_counts().sort_index().index,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    # User Input for Pairwise Scatter Plot
    st.subheader("Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis Feature", data.columns[:-2])
    y_axis = st.selectbox("Select Y-axis Feature", data.columns[:-2])
    color_feature = st.selectbox("Color By", ['quality'] + list(data.columns[:-2]))
    fig = px.scatter(
        data,
        x=x_axis,
        y=y_axis,
        color=color_feature,
        title=f"Scatter Plot: {y_axis} vs {x_axis}",
        labels={x_axis: x_axis.capitalize(), y_axis: y_axis.capitalize(), color_feature: color_feature.capitalize()},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    # Filter Data for Alcohol Content
    st.subheader("Filter Data by Alcohol Content")
    alcohol_range = st.slider("Select Alcohol Range (% vol)", 
                              min_value=float(data['alcohol'].min()), 
                              max_value=float(data['alcohol'].max()), 
                              value=(float(data['alcohol'].min()), float(data['alcohol'].max())))
    filtered_data = data[(data['alcohol'] >= alcohol_range[0]) & (data['alcohol'] <= alcohol_range[1])]
    st.write(f"Filtered Data: {filtered_data.shape[0]} samples")
    fig = px.histogram(
        filtered_data,
        x="alcohol",
        nbins=30,
        title=f"Alcohol Distribution (Range: {alcohol_range[0]} - {alcohol_range[1]} % vol)",
        color="quality",
        labels={"alcohol": "Alcohol (% vol)", "quality": "Wine Quality"},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig)

    # Boxplot for Selected Feature by Quality
    st.subheader("Boxplot Analysis")
    feature = st.selectbox("Select Feature for Boxplot", data.columns[:-2])
    fig = px.box(
        data,
        x="quality",
        y=feature,
        color="quality",
        title=f"Boxplot: {feature.capitalize()} by Quality",
        labels={"quality": "Wine Quality", feature: feature.capitalize()},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig)

    # User-Selected Heatmap
    st.subheader("Custom Correlation Heatmap")
    selected_features = st.multiselect(
        "Select Features for Heatmap", data.columns[:-2], default=list(data.columns[:-2])
    )
    if selected_features:
        corr = data[selected_features].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="Viridis",
                zmin=-1,
                zmax=1
            )
        )
        fig.update_layout(title="Feature Correlations", xaxis_title="Features", yaxis_title="Features")
        st.plotly_chart(fig)
    else:
        st.info("Select at least one feature to generate the heatmap.")

    # Fun Section: Predict Top Contributors
    st.subheader("🍷 Fun Insight: Alcohol and Quality")
    st.markdown("""
        Let's check if higher alcohol content leads to better wine quality! 🤔
    """)
    avg_quality = data.groupby("quality")["alcohol"].mean().reset_index()
    fig = px.line(
        avg_quality,
        x="quality",
        y="alcohol",
        title="Average Alcohol Content by Wine Quality",
        labels={"quality": "Wine Quality", "alcohol": "Average Alcohol (% vol)"},
        markers=True
    )
    st.plotly_chart(fig)

# Navigation
page = st.sidebar.radio("Navigate", ["Home", "Prediction", "Analysis"])

if page == "Home":
    home_page()
elif page == "Prediction":
    prediction_page()
elif page == "Analysis":
    analysis_page()