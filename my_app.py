# Streamlit ML Visualization App
# Advanced visualization and ML model showcase

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from openai import OpenAI
from langchain_community.document_loaders import PyPDFium2Loader
import tempfile

# Set page configuration
st.set_page_config(
    page_title="ML Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4267B2;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">📊 ML Visualization Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Explore data, visualizations, and ML predictions all in one place!</p>', unsafe_allow_html=True)

# Sidebar with navigation
st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png", width=200)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Introduction", "Data Explorer", "Visualization Suite", "ML Predictions", "Summarise"])

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Advertising.csv")
        return df
    except:
        # Generate sample data if file doesn't exist
        data = {
            'TV': np.random.uniform(10, 300, 200),
            'radio': np.random.uniform(1, 50, 200),
            'newspaper': np.random.uniform(0, 120, 200),
            'sales': np.random.uniform(1, 30, 200)
        }
        df = pd.DataFrame(data)
        return df

# Load or create model function
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("my_model", "rb"))
        return model
    except:
        # Create dummy model if file doesn't exist
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = df[['TV', 'radio', 'newspaper']]
        y = df['sales']
        model.fit(X, y)
        return model

df = load_data()

# Introduction page
if app_mode == "Introduction":
    st.markdown('<p class="sub-header">Welcome to the ML Visualization Dashboard!</p>', unsafe_allow_html=True)
    
    # Use columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This app showcases:
        - **Interactive Data Exploration**: Filter and examine data in real-time
        - **Advanced Visualizations**: Interactive plots and charts for better insights
        - **ML Model Predictions**: Test a trained model with custom inputs
        - **Performance Metrics**: Evaluate model performance with various metrics

        Use the sidebar to navigate between different sections of the app.
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1200/1*7BiqpKrBzc_6K1SnCGwgOA.png", width=300)
    
    # Create expander for quick tips
    with st.expander("Quick Tips"):
        st.markdown("""
        - 💡 Use the **Data Explorer** to understand the dataset
        - 📊 Create insightful visualizations in the **Visualization Suite**
        - 🧠 Test the ML model in the **ML Predictions** section
        - 🔍 Adjust parameters using sidebar sliders for real-time updates
        """)
    
    # Demo video or animation
    st.subheader("App Demo")
    st.video("https://www.youtube.com/watch?v=B2iAodr0fOo")

# Data Explorer page
elif app_mode == "Data Explorer":
    st.markdown('<p class="sub-header">Data Explorer</p>', unsafe_allow_html=True)
    
    # Data overview
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Add data filters
        with st.expander("Data Filters", expanded=True):
            # Create filter columns
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                min_tv = st.slider("Min TV Budget", float(df['TV'].min()), float(df['TV'].max()), float(df['TV'].min()))
            
            with filter_col2:
                min_radio = st.slider("Min Radio Budget", float(df['radio'].min()), float(df['radio'].max()), float(df['radio'].min()))
            
            with filter_col3:
                min_newspaper = st.slider("Min Newspaper Budget", float(df['newspaper'].min()), float(df['newspaper'].max()), float(df['newspaper'].min()))
        
        # Filter the dataframe
        filtered_df = df[(df['TV'] >= min_tv) & (df['radio'] >= min_radio) & (df['newspaper'] >= min_newspaper)]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, height=300)
    
    with col2:
        st.markdown('<p class="highlight">Data Summary</p>', unsafe_allow_html=True)
        st.write(f"**Total Entries:** {len(df)}")
        st.write(f"**Features:** {len(df.columns) - 1}")
        st.write(f"**Target:** sales")
        
        # Display missing values if any
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning("Missing Values Detected!")
            st.write(missing[missing > 0])
        else:
            st.success("No Missing Values!")
    
    # Show statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, height=300)
    
    # Show correlations with heatmap
    st.subheader("Correlation Heatmap")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Additional data insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Sales")
        fig = px.histogram(df, x="sales", nbins=20, title="Sales Distribution")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Budget Distribution")
        fig = px.box(df.melt(value_vars=["TV", "radio", "newspaper"]), 
                     x="variable", y="value", color="variable",
                     title="Distribution of Advertising Budgets")
        st.plotly_chart(fig, use_container_width=True)

# Visualization Suite page
elif app_mode == "Visualization Suite":
    st.markdown('<p class="sub-header">Visualization Suite</p>', unsafe_allow_html=True)
    
    viz_option = st.selectbox(
        "Choose Visualization Type",
        ["Scatter Plots", "Regression Analysis", "3D Visualization", "Feature Importance", "Custom Plot"]
    )
    
    if viz_option == "Scatter Plots":
        st.subheader("Relationship Between Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", df.columns.tolist())
        
        with col2:
            y_axis = st.selectbox("Y-axis", [col for col in df.columns if col != x_axis], index=0)
        
        color_option = st.checkbox("Add color dimension", value=True)
        if color_option:
            color_var = st.selectbox("Color Variable", [col for col in df.columns if col not in [x_axis, y_axis]])
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var, 
                             size_max=10, opacity=0.7,
                             trendline="ols", # Add trend line
                             title=f"{x_axis} vs {y_axis} colored by {color_var}")
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                             trendline="ols",
                             title=f"{x_axis} vs {y_axis}")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation stats
        st.metric("Correlation", round(df[x_axis].corr(df[y_axis]), 3))
    
    elif viz_option == "Regression Analysis":
        st.subheader("Advertising Budget vs Sales")
        
        feature = st.selectbox("Select Advertising Channel", ["TV", "radio", "newspaper"])
        
        fig = px.scatter(df, x=feature, y="sales", trendline="ols",
                        title=f"Impact of {feature} Advertising on Sales")
        
        # Get OLS regression results
        import statsmodels.api as sm
        X = sm.add_constant(df[feature])
        model = sm.OLS(df["sales"], X).fit()
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display regression stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R-squared", round(model.rsquared, 3))
        
        with col2:
            st.metric("Coefficient", round(model.params[feature], 3))
        
        with col3:
            st.metric("p-value", "{:.3e}".format(model.pvalues[feature]))
        
        # Display regression equation
        eq = f"Sales = {round(model.params['const'], 2)} + {round(model.params[feature], 2)} × {feature}"
        st.info(f"**Regression Equation:** {eq}")
        
        # Show summary table
        with st.expander("View Regression Summary"):
            st.text(model.summary().as_text())
    
    elif viz_option == "3D Visualization":
        st.subheader("3D Relationship Between Variables")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("X-axis Variable", ["TV", "radio", "newspaper"])
        
        with col2:
            y_var = st.selectbox("Y-axis Variable", [col for col in ["TV", "radio", "newspaper"] if col != x_var])
        
        with col3:
            z_var = st.selectbox("Z-axis Variable", ["sales"])
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df, x=x_var, y=y_var, z=z_var,
            color="sales", opacity=0.7,
            title=f"3D Plot of {x_var}, {y_var}, and {z_var}"
        )
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=z_var
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Feature Importance":
        st.subheader("Feature Importance Analysis")
        
        # Load model
        model = load_model()
        
        # Extract coefficients if it's a linear model
        try:
            # Calculate absolute importance
            importance = np.abs(model.coef_)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': ['TV', 'radio', 'newspaper'],
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                importance_df, 
                x='Feature', 
                y='Importance',
                color='Importance',
                title='Feature Importance (Based on Coefficient Magnitude)',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add normalized importance
            st.subheader("Relative Importance")
            
            # Normalize importance values for the pie chart
            importance_df['Relative'] = importance_df['Importance'] / importance_df['Importance'].sum()
            
            # Create pie chart for relative importance
            fig = px.pie(
                importance_df, 
                values='Relative', 
                names='Feature',
                title='Relative Feature Importance (%)',
                hole=0.4
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except:
            st.error("Could not extract feature importance from the model")
    
    elif viz_option == "Custom Plot":
        st.subheader("Build Your Own Visualization")
        
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Line Chart", "Bar Chart", "Bubble Chart", "Heatmap", "Violin Plot"]
        )
        
        if chart_type == "Line Chart":
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X-axis Variable (Line)", df.columns.tolist())
            
            with col2:
                y_vars = st.multiselect(
                    "Y-axis Variables (select multiple)",
                    [col for col in df.columns if col != x_var],
                    default=[col for col in df.columns if col != x_var][0]
                )
            
            if y_vars:
                # Sort the data by x_var for better line charts
                sorted_df = df.sort_values(by=x_var)
                
                fig = go.Figure()
                
                for y_var in y_vars:
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_df[x_var],
                            y=sorted_df[y_var],
                            mode='lines+markers',
                            name=y_var
                        )
                    )
                
                fig.update_layout(
                    title=f"Line Chart: {x_var} vs {', '.join(y_vars)}",
                    xaxis_title=x_var,
                    yaxis_title="Values",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one Y-axis variable")
        
        elif chart_type == "Bar Chart":
            # First, let's create some category groups for demonstration
            df['tv_category'] = pd.cut(df['TV'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                category_var = st.selectbox("Category Variable", ['tv_category'])
            
            with col2:
                value_var = st.selectbox("Value Variable", [col for col in df.columns if col not in ['tv_category']])
            
            agg_func = st.selectbox("Aggregation Function", ["Mean", "Sum", "Count", "Min", "Max"])
            
            # Map selected function to pandas agg function
            agg_map = {
                "Mean": "mean",
                "Sum": "sum",
                "Count": "count",
                "Min": "min",
                "Max": "max"
            }
            
            # Aggregate data
            agg_df = df.groupby(category_var)[value_var].agg(agg_map[agg_func]).reset_index()
            
            # Create bar chart
            fig = px.bar(
                agg_df,
                x=category_var,
                y=value_var,
                color=category_var,
                title=f"{agg_func} of {value_var} by {category_var}",
                labels={value_var: f"{agg_func} of {value_var}"}
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Bubble Chart":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                x_var = st.selectbox("X-axis (Bubble)", df.columns.tolist())
            
            with col2:
                y_var = st.selectbox("Y-axis (Bubble)", [col for col in df.columns if col != x_var])
            
            with col3:
                size_var = st.selectbox("Size Variable", [col for col in df.columns if col not in [x_var, y_var]])
            
            with col4:
                color_var = st.selectbox("Color Variable", [col for col in df.columns if col not in [x_var, y_var, size_var]])
            
            # Create bubble chart
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                size=size_var,
                color=color_var,
                hover_name=df.index,
                title=f"Bubble Chart of {x_var} vs {y_var}",
                labels={
                    x_var: x_var,
                    y_var: y_var,
                    size_var: size_var,
                    color_var: color_var
                }
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            st.write("This heatmap shows correlations between numerical features.")
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Violin Plot":
            st.write("Compare distributions with violin plots")
            
            # Create categories for demonstration
            df['sales_category'] = pd.qcut(df['sales'], q=3, labels=['Low Sales', 'Medium Sales', 'High Sales'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                category_var = st.selectbox("Category Variable (Violin)", ['sales_category'])
            
            with col2:
                value_vars = st.multiselect(
                    "Value Variables (select multiple for comparison)",
                    [col for col in df.columns if col not in ['sales_category']],
                    default=['TV', 'radio', 'newspaper']
                )
            
            if value_vars:
                # Melt the dataframe for easier plotting
                melted_df = df.melt(id_vars=[category_var], value_vars=value_vars,
                                    var_name='variable', value_name='value')
                
                # Create violin plot
                fig = px.violin(
                    melted_df,
                    x='variable',
                    y='value',
                    color=category_var,
                    box=True,
                    points="all",
                    title=f"Distribution of Variables by {category_var}"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one variable for comparison")

# ML Predictions page
elif app_mode == "ML Predictions":
    st.markdown('<p class="sub-header">Machine Learning Predictions</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Create tabs for different prediction modes
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Predictions", "Model Performance"])
    
    with tab1:
        st.markdown('<p class="highlight">Make a single prediction by adjusting the input parameters</p>', unsafe_allow_html=True)
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tv = st.slider("TV Advertising Budget", 
                          min_value=float(df['TV'].min()), 
                          max_value=float(df['TV'].max()),
                          value=float(df['TV'].median()),
                          step=5.0)
        
        with col2:
            radio = st.slider("Radio Advertising Budget", 
                             min_value=float(df['radio'].min()), 
                             max_value=float(df['radio'].max()),
                             value=float(df['radio'].median()),
                             step=1.0)
        
        with col3:
            newspaper = st.slider("Newspaper Advertising Budget", 
                                 min_value=float(df['newspaper'].min()), 
                                 max_value=float(df['newspaper'].max()),
                                 value=float(df['newspaper'].median()),
                                 step=1.0)
        
        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'TV': [tv],
            'radio': [radio],
            'newspaper': [newspaper]
        })
        
        # Make prediction
        if st.button("Predict Sales"):
            prediction = model.predict(input_data)[0]
            
            st.success(f"Predicted Sales: ${prediction:.2f}k")
            
            # Create a gauge chart for the prediction
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Sales ($k)"},
                gauge={
                    'axis': {'range': [df['sales'].min(), df['sales'].max()]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [df['sales'].min(), df['sales'].quantile(0.33)], 'color': "#ffdd99"},
                        {'range': [df['sales'].quantile(0.33), df['sales'].quantile(0.66)], 'color': "#90CAF9"},
                        {'range': [df['sales'].quantile(0.66), df['sales'].max()], 'color': "#6dd5ed"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': prediction
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show contribution of each feature
            st.subheader("Feature Contributions")
            
            # Calculate contributions (simplified for linear models)
            try:
                contributions = {}
                for i, feature in enumerate(['TV', 'radio', 'newspaper']):
                    contributions[feature] = model.coef_[i] * input_data[feature].values[0]
                
                # Add intercept
                contributions['Baseline'] = model.intercept_
                
                # Create dataframe for visualization
                contrib_df = pd.DataFrame({
                    'Feature': list(contributions.keys()),
                    'Contribution': list(contributions.values())
                })
                
                # Sort by absolute contribution
                contrib_df = contrib_df.sort_values('Contribution', key=abs, ascending=False)
                
                # Create waterfall chart
                fig = go.Figure(go.Waterfall(
                    name="Feature Contributions",
                    orientation="v",
                    measure=["relative"] * (len(contrib_df) - 1) + ["total"],
                    x=contrib_df['Feature'],
                    y=contrib_df['Contribution'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Sales Prediction Breakdown",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except:
                st.info("Feature contribution breakdown is only available linear models or models with feature importance (e.g. decision trees)")
            
            # Show similar instances from the dataset
            st.subheader("Similar Instances in Dataset")
            
            # Calculate Euclidean distance to find similar instances
            from scipy.spatial.distance import cdist
            
            # Normalize features for distance calculation
            scaler = StandardScaler()
            features = ['TV', 'radio', 'newspaper']
            
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df[features]),
                columns=features
            )
            
            input_scaled = pd.DataFrame(
                scaler.transform(input_data[features]),
                columns=features
            )
            
            # Calculate distances
            distances = cdist(input_scaled, df_scaled, 'euclidean')[0]
            
            # Get indices of 5 most similar instances
            similar_indices = distances.argsort()[:5]
            
            # Show similar instances
            similar_df = df.iloc[similar_indices].copy()
            similar_df['similarity'] = 1 / (1 + distances[similar_indices])
            similar_df['similarity'] = (similar_df['similarity'] * 100).round(2)
            
            st.dataframe(similar_df[['TV', 'radio', 'newspaper', 'sales', 'similarity']].sort_values('similarity', ascending=False))
    
    with tab2:
        st.markdown('<p class="highlight">Make predictions for multiple scenarios at once</p>', unsafe_allow_html=True)
        
        st.info("Specify ranges for each feature to create multiple scenarios")
        
        # Create columns for input ranges
        col1, col2 = st.columns(2)
        
        with col1:
            tv_range = st.slider("TV Budget Range", 
                               float(df['TV'].min()), 
                               float(df['TV'].max()), 
                               (float(df['TV'].quantile(0.25)), float(df['TV'].quantile(0.75))),
                               step=10.0)
            
            radio_range = st.slider("Radio Budget Range", 
                                  float(df['radio'].min()), 
                                  float(df['radio'].max()), 
                                  (float(df['radio'].quantile(0.25)), float(df['radio'].quantile(0.75))),
                                  step=5.0)
        
        with col2:
            newspaper_range = st.slider("Newspaper Budget Range", 
                                      float(df['newspaper'].min()), 
                                      float(df['newspaper'].max()), 
                                      (float(df['newspaper'].quantile(0.25)), float(df['newspaper'].quantile(0.75))),
                                      step=5.0)
            
            num_scenarios = st.number_input("Number of Scenarios", min_value=5, max_value=100, value=10, step=5)
        
        if st.button("Generate Scenarios"):
            # Generate random scenarios within the specified ranges
            tv_values = np.random.uniform(tv_range[0], tv_range[1], num_scenarios)
            radio_values = np.random.uniform(radio_range[0], radio_range[1], num_scenarios)
            newspaper_values = np.random.uniform(newspaper_range[0], newspaper_range[1], num_scenarios)
            
            # Create scenarios dataframe
            scenarios_df = pd.DataFrame({
                'TV': tv_values,
                'radio': radio_values,
                'newspaper': newspaper_values
            })
            
            # Make predictions
            scenarios_df['predicted_sales'] = model.predict(scenarios_df)
            
            # Round values for display
            scenarios_df = scenarios_df.round(2)
            
            # Add scenario ID
            scenarios_df['scenario_id'] = [f"Scenario {i+1}" for i in range(num_scenarios)]
            
            # Show results
            st.subheader("Prediction Results")
            st.dataframe(scenarios_df[['scenario_id', 'TV', 'radio', 'newspaper', 'predicted_sales']])
            
            # Visualize scenarios
            st.subheader("Scenario Comparison")
            
            # Sort by predicted sales
            sorted_scenarios = scenarios_df.sort_values('predicted_sales', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                sorted_scenarios,
                x='predicted_sales',
                y='scenario_id',
                orientation='h',
                color='predicted_sales',
                color_continuous_scale='Viridis',
                title="Scenarios Ranked by Predicted Sales"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bubble chart to compare scenarios
            fig = px.scatter(
                scenarios_df,
                x='TV',
                y='radio',
                size='predicted_sales',
                color='newspaper',
                hover_name='scenario_id',
                title="Scenario Comparison (Bubble Size = Predicted Sales)",
                labels={'newspaper': 'Newspaper Budget'},
                size_max=30
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimal scenario
            best_scenario = scenarios_df.loc[scenarios_df['predicted_sales'].idxmax()]
    with tab3:
        st.markdown('<p class="highlight">Include code here to display the model performance</p>')
elif app_mode == "Summarise": 
    # Create a section for OpenAI interaction
    st.header("OpenAI GPT-4o-mini Integration")

    # API Key input (you might want to use st.secrets in production)
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
    
        # File uploader for PDF documents
        uploaded_file = st.file_uploader("Upload a PDF file for analysis", type=["pdf"])
    
        # Function to read PDF document
        def read_doc(file_path):
            file_loader = PyPDFium2Loader(file_path)
            pdf_documents = file_loader.load()  # PyPDFium2Loader reads page by page
            return pdf_documents
    
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
        
            # Read the PDF
            try:
                pdf_documents = read_doc(temp_file_path)
                st.success(f"Successfully loaded PDF with {len(pdf_documents)} pages")
            
                # Extract text from the PDF
                pdf_text = ""
                for doc in pdf_documents:
                    pdf_text += doc.page_content + "\n\n"
            
                # Show a preview of the text
                with st.expander("PDF Content Preview"):
                    st.text(pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text)
            
                # User input for questions about the document
                st.subheader("Ask Questions About Your Document")
                user_question = st.text_area("What would you like to know about this document?", 
                                      "Summarize the key points of this document.")
            
                if st.button("Get AI Response"):
                    with st.spinner("Processing your request..."):
                        try:
                            # Create a chat completion with the document context
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                 {"role": "system", "content": "You are a helpful assistant specialized in analyzing documents. Provide concise and informative responses."},
                                 {"role": "user", "content": f"Here is a document to analyze:\n\n{pdf_text[:4000]}...\n\nQuestion: {user_question}"}
                                ],
                                temperature=0.0,
                                top_p=1
                            )
                        
                            # Display the response
                            st.subheader("AI Response")
                            st.write(response.choices[0].message.content)
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading the PDF: {str(e)}")
        
            # Clean up the temporary file
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
        # General chat interface (without document)
        st.subheader("General Chat")
        chat_input = st.text_area("Chat with GPT-4o-mini", "Hello, I have a question about...")
    
        if st.button("Send Message"):
            with st.spinner("Thinking..."):
                try:
                    # Create a chat completion
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": chat_input}
                        ],
                        temperature=0.0,
                        top_p=1
                    )
                
                    # Display the response
                    st.subheader("AI Response")
                    st.write(response.choices[0].message.content)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your OpenAI API key to use this feature.")
