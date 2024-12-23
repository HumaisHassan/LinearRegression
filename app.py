import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# App title
st.title("Linear Regression Visualization App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:", data.head())
    
    # Select features for regression
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_columns) < 2:
        st.error("Dataset must have at least two numeric columns.")
    else:
        x_col = st.selectbox("Select the feature (X)", numeric_columns)
        y_col = st.selectbox("Select the target (Y)", numeric_columns)

        if x_col and y_col:
            # Prepare data
            X = data[[x_col]].values
            y = data[y_col].values

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(X, y, color="blue", label="Data Points")
            ax.plot(X, model.predict(X), color="red", label="Regression Line")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Linear Regression: {y_col} vs {x_col}")
            ax.legend()

            # Display plot
            st.pyplot(fig)

            # Display metrics
            st.write(f"**Model Coefficient (Slope):** {model.coef_[0]:.2f}")
            st.write(f"**Model Intercept:** {model.intercept_:.2f}")
            st.write(f"**R² Score:** {model.score(X_test, y_test):.2f}")
