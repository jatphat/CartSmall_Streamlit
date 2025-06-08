# import streamlit as st
"""

CartSmall - Waste Reduction Marketplace Streamlit Application
This application provides a comprehensive dashboard and predictive analytics platform for waste reduction activities, focusing on transactions such as selling, buying, exchanging, gifting, wishing, and composting items. The app is designed for individual users, apartments, counties, and states, offering both historical insights and advanced forecasting using machine learning models.
Main Features:
---------------
- User Dashboard: Visualizes individual user's transaction history, key metrics, and environmental impact.
- Transaction Form: Allows users to log new transactions with dynamic fields based on transaction type.
- Apartment & County Dashboards: Aggregated statistics and visualizations for apartments and counties, including monthly trends and category breakdowns.
- Predictions Dashboard: Multi-level (overall, state, county, apartment) waste reduction forecasting using Random Forest, XGBoost, LightGBM, and Gradient Boosting models with cross-validation and feature importance analysis.
- Model Performance: Detailed metrics (R², RMSE, MAE, MAPE), error analysis, and feature importance visualizations for model interpretability.
- Data Validation: Ensures sufficient data is available for reliable predictions at each aggregation level.
- Robust Error Handling: User-friendly error messages and debug information for troubleshooting.
Key Functions:
--------------
- load_actual_data: Loads and preprocesses the unified dataset, extracting user and transaction information.
- show_dashboard: Displays the user's impact dashboard with historical data and environmental metrics.
- show_transaction_form: Presents a dynamic form for logging new transactions.
- train_level_specific_model: Trains and compares multiple regression models with enhanced feature engineering and cross-validation.
- show_prediction_plot: Visualizes historical and predicted waste reduction with confidence intervals.
- show_model_performance_metrics: Calculates and displays model evaluation metrics and quality assessment.
- show_predictions_dashboard: Main entry point for multi-level prediction analysis.
- show_county_dashboard, show_apartment_dashboard: Aggregated dashboards for county and apartment performance.
- Utility functions for feature engineering, error analysis, and visualization.
Dependencies:
-------------
- streamlit, pandas, numpy, plotly, scikit-learn, xgboost, lightgbm
Usage:
------
Run the script as a Streamlit app. Users can select their profile, view dashboards, log transactions, and explore predictive analytics at various aggregation levels.

...
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set page configuration
st.set_page_config(
    page_title="CartSmall - Waste Reduction Marketplace",
    page_icon="♻️",
    layout="wide"
)

@st.cache_data
def load_actual_data():
    try:
        df = pd.read_csv('Unified_Dataset_Final.csv')
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df['Seller_DOB'] = pd.to_datetime(df['Seller_DOB'])

        sellers = df[['Seller_ID', 'Seller_First_Name', 'Seller_Last_Name',
                      'Seller_Apartment', 'Seller_County', 'Seller_State']].drop_duplicates()
        sellers.columns = ['user_id', 'first_name', 'last_name', 'apartment', 'county', 'state']

        users_df = sellers.copy()
        users_df['full_name'] = users_df['first_name'] + ' ' + users_df['last_name']

        transactions_df = df[
            [
                'Transaction_ID', 'Seller_ID', 'Item_Name', 'Category', 'Weight',
                'Selling_Price', 'Cost_Price', 'Condition', 'Transaction_Type',
                'Transaction_Date', 'Seller_Apartment', 'Seller_County', 'Seller_State'
            ]
        ].copy()

        return users_df, transactions_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def validate_data_for_prediction(data, level):
    """Validate if there's enough data for prediction"""
    min_records = 30  # Minimum number of records needed
    if len(data) < min_records:
        st.warning(
            f"Limited data available for {level} predictions. Results may be less accurate."
        )
        return False
    return True

def show_dashboard(selected_user, users_df, transactions_df):
    """Show individual user dashboard with historical data only"""
    if selected_user is None:
        st.warning("No user selected")
        return

    # Get user's transactions
    user_transactions = transactions_df[transactions_df['Seller_ID'] == selected_user]

    st.subheader("Your Impact Dashboard")

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(user_transactions))
    with col2:
        st.metric(
            "Total Weight Processed",
            f"{user_transactions['Weight'].sum():.1f} kg"
        )
    with col3:
        total_value = user_transactions['Selling_Price'].sum()
        st.metric("Total Value", f"${total_value:.2f}")

    if not user_transactions.empty:
        # Transaction History Chart (renamed for clarity)
        st.subheader("Your Waste Reduction History")
        daily_data = (
            user_transactions.groupby('Transaction_Date')['Weight']
            .sum().reset_index()
        )
        fig_history = px.line(
            daily_data,
            x='Transaction_Date',
            y='Weight',
            title='Historical Daily Waste Reduction'
        )
        st.plotly_chart(fig_history, use_container_width=True)

        # Transaction Type Distribution
        st.subheader("Transaction Types")
        fig_types = px.pie(
            user_transactions,
            names='Transaction_Type',
            title='Your Transaction Distribution'
        )
        st.plotly_chart(fig_types, use_container_width=True)

        # Category Analysis
        st.subheader("Categories")
        category_data = (
            user_transactions.groupby('Category')['Weight']
            .sum().reset_index()
        )
        fig_categories = px.bar(
            category_data,
            x='Category',
            y='Weight',
            title='Weight by Category'
        )
        st.plotly_chart(fig_categories, use_container_width=True)

        # Recent Transactions
        st.subheader("Recent Transactions")
        recent = user_transactions.sort_values('Transaction_Date', ascending=False).head(10)
        st.dataframe(recent)

        # Environmental Impact
        st.subheader("Environmental Impact")
        total_weight = user_transactions['Weight'].sum()
        co2_per_kg = 1.8  # Example value
        water_per_kg = 2500  # Example value
        trees_per_kg = 0.1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CO₂ Saved", f"{total_weight * co2_per_kg:.1f} kg")
        with col2:
            st.metric("Water Saved", f"{total_weight * water_per_kg:.0f} L")
        with col3:
            st.metric("Trees Equivalent", f"{total_weight * trees_per_kg:.1f}")

def show_transaction_form():
    """Display dynamic transaction form"""
    st.subheader("New Transaction")

    transaction_type = st.selectbox(
        "What would you like to do?",
        ["Select Action", "Sell", "Buy", "Exchange", "Gift", "Wish", "Compost"]
    )

    if transaction_type != "Select Action":
        with st.form(key=f"{transaction_type}_form"):
            st.write(f"New {transaction_type} Transaction")

            # Common fields
            st.text_input("Item Name")
            st.selectbox(
                "Category",
                ["Food", "Clothing", "Electronics", "Furniture", "Books", "Other"]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Weight (kg)", min_value=0.0, step=0.1)

            # Dynamic fields
            if transaction_type in ["Sell", "Buy"]:
                with col2:
                    st.number_input("Price ($)", min_value=0.0)
                st.selectbox(
                    "Item Condition",
                    ["Excellent", "Good", "Fair", "Poor"]
                )

            if transaction_type == "Exchange":
                st.text_input("What would you like in exchange?")

            if transaction_type == "Wish":
                st.selectbox(
                    "Urgency Level",
                    ["Low", "Medium", "High"]
                )

            if transaction_type == "Compost":
                st.selectbox(
                    "Food Waste Type",
                    ["Fruits & Vegetables", "Coffee Grounds", "Tea Bags", "Other"]
                )

            st.text_area("Description")

            submitted = st.form_submit_button("Submit Transaction")
            if submitted:
                st.success(f"{transaction_type} transaction submitted successfully!")

def train_level_specific_model(data, level_column=None):
    """Enhanced model training with multiple models, cross-validation, and regularization"""
    with st.spinner('Training models... This may take a few moments.'):
        df = data.copy()
    try:
        # 1. Enhanced Feature Engineering
        df = create_features(df, level_column)

        # 2. Prepare Data
        X, y, features = prepare_data(df, level_column)

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. Define models with regularization
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                enable_categorical=True
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            )
        }

        # 6. Cross-validation setup
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # 7. Train and evaluate models
        results = []
        best_score = -np.inf
        best_model = None
        best_predictions = None

        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=kfold,
                scoring='r2'
            )

            if name == 'LightGBM':
                model.set_params(feature_name=features)

            # Train model
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)

            # Calculate MAPE
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

            results.append({
                'Model': name,
                'R2 Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'CV Mean R2': cv_scores.mean(),
                'CV Std R2': cv_scores.std()
            })

            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_predictions = predictions

        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Model Comparison")
        st.dataframe(results_df.style.format({
            'R2 Score': '{:.3f}',
            'RMSE': '{:.3f}',
            'MAE': '{:.3f}',
            'MAPE': '{:.1f}%',
            'CV Mean R2': '{:.3f}',
            'CV Std R2': '{:.3f}'
        }))

        # Feature importance for best model
        if hasattr(best_model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.subheader("Feature Importance (Best Model)")
            fig = px.bar(
                feature_imp.head(15),
                x='Feature',
                y='Importance',
                title='Top 15 Most Important Features'
            )
            st.plotly_chart(fig)

        return best_model, best_predictions, y_test, features

    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None

def create_features(df, level_column):
    """Enhanced feature engineering"""
    # Time-based features
    df['year'] = df['Transaction_Date'].dt.year
    df['month'] = df['Transaction_Date'].dt.month
    df['day_of_week'] = df['Transaction_Date'].dt.dayofweek
    df['day_of_month'] = df['Transaction_Date'].dt.day
    df['week_of_year'] = df['Transaction_Date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12], labels=['Winter','Spring','Summer','Fall'])

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

    # Rolling statistics
    group_col = level_column if level_column else 'Seller_ID'
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}d'] = df.groupby(group_col)['Weight'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}d'] = df.groupby(group_col)['Weight'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f'rolling_max_{window}d'] = df.groupby(group_col)['Weight'].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )

    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'weight_lag_{lag}'] = df.groupby(group_col)['Weight'].shift(lag)

    # Transaction features
    df['price_per_weight'] = df['Selling_Price'] / df['Weight'].clip(lower=0.01)
    df['total_value'] = df['Weight'] * df['Selling_Price']

    return df

def prepare_data(df, level_column):
    """Prepare data for modeling"""
    try:
        # Encode categorical variables
        le = LabelEncoder()
        df['Category_encoded'] = le.fit_transform(df['Category'])
        df['Transaction_Type_encoded'] = le.fit_transform(df['Transaction_Type'])

        if level_column:
            df[f'{level_column}_encoded'] = le.fit_transform(df[level_column])

        # Select features with explicit names
        feature_patterns = [
            'encoded', 'rolling', 'lag', 'sin', 'cos', 'is_weekend',
            'price_per_weight', 'total_value', 'year', 'month', 'day'
        ]

        # Get feature names explicitly
        features = []
        for pattern in feature_patterns:
            features.extend([col for col in df.columns if pattern in col])

        # Ensure features exist in dataframe
        features = [f for f in features if f in df.columns]

        # Create X with named features
        X = pd.DataFrame(
            df[features].fillna(method='ffill').fillna(method='bfill'),
            columns=features  # Explicitly set column names
        )
        y = df['Weight']

        return X, y, features

    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None

def show_prediction_plot(model, features, future_dates, historical_data, plot_title):
    """Show prediction plot"""
    try:
        # Create future features DataFrame
        future_df = pd.DataFrame({
            'Transaction_Date': future_dates,
            'year': future_dates.year,
            'month': future_dates.month,
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'week_of_year': future_dates.isocalendar().week,
            'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int)
        })

        # Add missing features with default values
        for feature in features:
            if feature not in future_df.columns:
                if 'encoded' in feature:
                    future_df[feature] = 0  # Default encoding
                elif 'rolling' in feature:
                    future_df[feature] = historical_data['Weight'].mean()  # Use historical mean
                elif 'price' in feature:
                    future_df[feature] = historical_data['Selling_Price'].mean()  # Use historical mean
                else:
                    future_df[feature] = 0

        # Make predictions
        future_predictions = model.predict(future_df[features])

        # Create plot
        fig = go.Figure()

        # Historical data
        historical = historical_data.groupby('Transaction_Date')['Weight'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=historical['Transaction_Date'],
            y=historical['Weight'],
            name='Historical',
            line=dict(color='blue')
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))

        # Add confidence intervals (using standard deviation of historical data)
        std_dev = historical['Weight'].std()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions + 1.96 * std_dev,
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Upper CI'
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions - 1.96 * std_dev,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Lower CI'
        ))

        # Update layout
        fig.update_layout(
            title=f'Waste Reduction Forecast - {plot_title}',
            xaxis_title='Date',
            yaxis_title='Weight (kg)',
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error in prediction plot: {str(e)}")

def show_overall_predictions(transactions_df):
    """Show overall predictions with detailed metrics"""
    if validate_data_for_prediction(transactions_df, "overall"):
        model, predictions, y_test, features = train_level_specific_model(transactions_df)

        if model is not None:
            # Show model performance metrics
            show_model_performance_metrics(y_test, predictions, "Overall Model")

            # Feature importance
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_imp = px.bar(
                feature_imp,
                x='Feature',
                y='Importance',
                title='Feature Importance in Predictions'
            )
            st.plotly_chart(fig_imp)

            # Future predictions
            future_dates = pd.date_range(
                start=transactions_df['Transaction_Date'].max(),
                periods=30,
                freq='D'
            )

            show_prediction_plot(model, features, future_dates, transactions_df, "Overall")

def show_model_performance_metrics(y_test, predictions, level_name=""):
    """Display detailed model performance metrics with improved handling and scaling"""
    st.subheader(f"Model Performance Metrics for {level_name}")

    try:
        # Basic calculations
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = np.mean(np.abs(predictions - y_test))

        # MAPE calculation with improved handling
        non_zero_mask = y_test != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) /
                                  np.abs(y_test[non_zero_mask]))) * 100
        else:
            mape = 0

        # R² calculation with improved handling
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Remove units from display values
        rmse_display = f"{rmse:.2f}"
        mae_display = f"{mae:.2f}"

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", rmse_display)
        with col2:
            st.metric("MAE", mae_display)
        with col3:
            st.metric("MAPE", f"{min(mape, 100):.1f}%")  # Cap MAPE at 100%
        with col4:
            st.metric("R² Score", f"{r2:.3f}")

        # Add interpretation
        st.info("""
        📊 Metrics Interpretation:
        - RMSE: Root Mean Square Error (lower is better)
        - MAE: Mean Absolute Error (lower is better)
        - MAPE: Mean Absolute Percentage Error (lower is better)
        - R² Score: Coefficient of Determination (higher is better, max 1.0)
        """)

        # Evaluate model quality using multiple metrics
        quality_metrics = {
            'rmse_ratio': rmse / np.mean(y_test),
            'mape': min(mape, 100) / 100,
            'r2': max(0, r2)  # Convert negative R² to 0
        }

        # Calculate overall quality score (0 to 1)
        quality_score = (
            (1 - quality_metrics['rmse_ratio']) * 0.4 +
            (1 - quality_metrics['mape']) * 0.3 +
            quality_metrics['r2'] * 0.3
        )

        # Display model quality assessment
        if quality_score > 0.7:
            st.success("✅ Model Quality: Excellent - Predictions are highly reliable")
        elif quality_score > 0.5:
            st.success("✅ Model Quality: Good - Predictions are reliable")
        elif quality_score > 0.3:
            st.warning("⚠️ Model Quality: Fair - Predictions should be used with caution")
        else:
            st.warning("""
            ⚠️ Model Quality: Needs Improvement

            Suggestions for improvement:
            1. Consider collecting more data points
            2. Check for and handle outliers
            3. Add more relevant features
            4. Try different time periods
            """)

        # Show relative error distribution
        st.subheader("Prediction Error Distribution")
        error_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions,
            'Error': np.abs(y_test - predictions) / y_test * 100
        })

        error_ranges = {
            '< 10%': (error_df['Error'] < 10).mean() * 100,
            '10-25%': ((error_df['Error'] >= 10) & (error_df['Error'] < 25)).mean() * 100,
            '25-50%': ((error_df['Error'] >= 25) & (error_df['Error'] < 50)).mean() * 100,
            '> 50%': (error_df['Error'] >= 50).mean() * 100
        }

        error_ranges_df = pd.DataFrame({
            'Range': list(error_ranges.keys()),
            'Percentage of Predictions': [f"{v:.1f}%" for v in error_ranges.values()]
        })
        st.dataframe(error_ranges_df)

    except Exception as e:
        st.error(f"Error in model performance metrics: {str(e)}")

def show_state_predictions(transactions_df):
    st.info("State-level predictions not implemented in this code snippet.")

def show_county_predictions(transactions_df):
    st.info("County-level predictions not implemented in this code snippet.")

def show_apartment_predictions(transactions_df):
    """Show apartment-level predictions"""
    selected_state = st.selectbox(
        "Select State",
        sorted(transactions_df['Seller_State'].unique()),
        key="apt_pred_state_select"
    )

    state_apartments = transactions_df[
        transactions_df['Seller_State'] == selected_state
    ]['Seller_Apartment'].unique()

    selected_apartment = st.selectbox(
        "Select Apartment",
        sorted(state_apartments),
        key="apt_pred_apt_select"
    )

    apt_data = transactions_df[transactions_df['Seller_Apartment'] == selected_apartment]

    if validate_data_for_prediction(apt_data, "apartment"):
        model, predictions, y_test, features = train_level_specific_model(apt_data, 'Seller_Apartment')

        if model is not None:
            show_model_performance_metrics(y_test, predictions, f"Apartment: {selected_apartment}")

            # Feature importance
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_imp = px.bar(
                feature_imp,
                x='Feature',
                y='Importance',
                title=f'Feature Importance for {selected_apartment}'
            )
            st.plotly_chart(fig_imp)

            # Future predictions
            future_dates = pd.date_range(
                start=apt_data['Transaction_Date'].max(),
                periods=30,
                freq='D'
            )

            show_prediction_plot(model, features, future_dates, apt_data,
                                f"Apartment: {selected_apartment}")

def calculate_detailed_metrics(y_true, y_pred):
    """
    Calculate comprehensive model performance metrics
    """
    metrics = {}

    # Basic metrics
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # MAPE with handling for zero values
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                                y_true[non_zero_mask])) * 100
    else:
        metrics['mape'] = 0

    return metrics

def create_comparison_plot(y_true, y_pred):
    """
    Create enhanced actual vs predicted comparison plot
    """
    fig = go.Figure()

    # Scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        )
    ))

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True,
        width=800,
        height=600
    )

    return fig

def show_feature_importance(model, features):
    """
    Display detailed feature importance analysis
    """
    st.subheader("Feature Importance Analysis")

    # Get feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    fig = px.bar(
        feature_imp,
        x='Feature',
        y='Importance',
        title='Feature Importance Ranking'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )

    st.plotly_chart(fig)

    # Show top features table
    st.write("Top 10 Most Important Features:")
    st.dataframe(
        feature_imp.head(10).style.format({'Importance': '{:.4f}'})
    )

def show_error_analysis(y_true, y_pred):
    """
    Display comprehensive error analysis
    """
    st.subheader("Error Analysis")

    # Calculate errors
    errors = y_true - y_pred
    percentage_errors = (errors / y_true) * 100

    # Error distribution plot
    fig_error = go.Figure()
    fig_error.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        name='Error Distribution'
    ))

    fig_error.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Error',
        yaxis_title='Count'
    )

    st.plotly_chart(fig_error)

    # Error statistics
    col1, col2 = st.columns(2)
    with col1:
        st.write("Error Statistics:")
        error_stats = pd.DataFrame({
            'Metric': ['Mean Error', 'Std Error', 'Median Error'],
            'Value': [
                f"{errors.mean():.2f}",
                f"{errors.std():.2f}",
                f"{np.median(errors):.2f}"
            ]
        })
        st.dataframe(error_stats)

    with col2:
        st.write("Error Percentiles:")
        percentiles = [10, 25, 50, 75, 90]
        error_percentiles = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Value': [f"{np.percentile(errors, p):.2f}" for p in percentiles]
        })
        st.dataframe(error_percentiles)

    # Error ranges
    st.write("Prediction Error Ranges:")
    error_ranges = {
        '< 10%': (np.abs(percentage_errors) < 10).mean() * 100,
        '10-25%': ((np.abs(percentage_errors) >= 10) &
                   (np.abs(percentage_errors) < 25)).mean() * 100,
        '25-50%': ((np.abs(percentage_errors) >= 25) &
                   (np.abs(percentage_errors) < 50)).mean() * 100,
        '> 50%': (np.abs(percentage_errors) >= 50).mean() * 100
    }

    error_ranges_df = pd.DataFrame({
        'Range': list(error_ranges.keys()),
        'Percentage of Predictions': [f"{v:.1f}%" for v in error_ranges.values()]
    })
    st.dataframe(error_ranges_df)

def show_predictions_dashboard(transactions_df):
    """Enhanced predictions dashboard with multi-level forecasting"""
    st.header("Waste Reduction Predictions")

    try:
        prediction_level = st.selectbox(
            "Select Prediction Level",
            ["Overall", "State", "County", "Apartment"],
            key="prediction_level_main"
        )

        if prediction_level == "Overall":
            show_overall_predictions(transactions_df)
        elif prediction_level == "State":
            show_state_predictions(transactions_df)
        elif prediction_level == "County":
            show_county_predictions(transactions_df)
        else:
            show_apartment_predictions(transactions_df)

    except Exception as e:
        st.error(f"Error in predictions: {str(e)}")
        st.info("This might be due to insufficient data for the selected level.")

def show_county_dashboard(transactions_df):
    """Show enhanced county-level dashboard with state filtering"""
    st.header("County Performance Dashboard")

    try:
        # State filter
        states = sorted(transactions_df['Seller_State'].unique())
        selected_state = st.selectbox("Select State", states, key='county_state_select')

        # Filter transactions by state
        state_transactions = transactions_df[transactions_df['Seller_State'] == selected_state]

        # Transaction type filter
        transaction_types = ['All'] + sorted(state_transactions['Transaction_Type'].unique())
        selected_type = st.selectbox("Transaction Type", transaction_types, key='county_type_select')

        if selected_type != 'All':
            state_transactions = state_transactions[state_transactions['Transaction_Type'] == selected_type]

        # Group by county
        county_stats = state_transactions.groupby('Seller_County').agg({
            'Weight': 'sum',
            'Transaction_ID': 'count',
            'Seller_Apartment': 'nunique'
        }).reset_index()

        county_stats.columns = ['County', 'Total Weight (kg)', 'Transactions', 'Active Apartments']
        county_stats = county_stats.sort_values('Total Weight (kg)', ascending=False)

        # Display metrics
        st.subheader(f"{selected_state} Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Counties", len(county_stats))
        with col2:
            st.metric("Total Weight Reduced", f"{county_stats['Total Weight (kg)'].sum():.1f} kg")
        with col3:
            st.metric("Total Transactions", f"{county_stats['Transactions'].sum():,}")

        # County comparison chart
        st.subheader("County Comparison")
        fig_counties = px.bar(
            county_stats,
            x='County',
            y='Total Weight (kg)',
            title=f'Waste Reduction by County in {selected_state}',
            color='Total Weight (kg)',
            hover_data=['Transactions', 'Active Apartments']
        )
        st.plotly_chart(fig_counties)

        # Monthly trends by county
        st.subheader("Monthly Trends by County")
        monthly_county = state_transactions.groupby([
            'Seller_County',
            pd.Grouper(key='Transaction_Date', freq='M')
        ])['Weight'].sum().reset_index()

        fig_trends = px.line(
            monthly_county,
            x='Transaction_Date',
            y='Weight',
            color='Seller_County',
            title=f'Monthly Waste Reduction Trends by County in {selected_state}'
        )
        st.plotly_chart(fig_trends)

        # Transaction type distribution by county
        if selected_type == 'All':
            st.subheader("Transaction Types by County")
            county_types = state_transactions.groupby(['Seller_County', 'Transaction_Type'])['Weight'].sum().reset_index()
            fig_types = px.bar(
                county_types,
                x='Seller_County',
                y='Weight',
                color='Transaction_Type',
                title=f'Transaction Types Distribution by County in {selected_state}',
                barmode='stack'
            )
            st.plotly_chart(fig_types)

        # Top performing counties table
        st.subheader("County Rankings")
        st.dataframe(
            county_stats.style.format({
                'Total Weight (kg)': '{:.1f}',
                'Transactions': '{:,}',
                'Active Apartments': '{:,}'
            })
        )

    except Exception as e:
        st.error(f"Error in county dashboard: {str(e)}")
        st.info("There might be an issue with the data or filtering.")

def show_apartment_dashboard(apartment_id, transactions_df):
    """Show detailed apartment dashboard"""
    st.header(f"Apartment {apartment_id} Overview")

    apt_transactions = transactions_df[transactions_df['Seller_Apartment'] == apartment_id]

    if not apt_transactions.empty:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Weight Reduced",
                     f"{apt_transactions['Weight'].sum():.1f} kg")
        with col2:
            st.metric("Total Transactions",
                     len(apt_transactions))
        with col3:
            st.metric("Average Transaction Size",
                     f"{apt_transactions['Weight'].mean():.1f} kg")

        # Monthly Progress
        st.subheader("Monthly Progress")
        monthly_data = apt_transactions.groupby(
            apt_transactions['Transaction_Date'].dt.to_period('M')
        )['Weight'].sum().reset_index()
        monthly_data['Transaction_Date'] = monthly_data['Transaction_Date'].astype(str)

        fig_monthly = px.line(
            monthly_data,
            x='Transaction_Date',
            y='Weight',
            title='Monthly Waste Reduction'
        )
        st.plotly_chart(fig_monthly)

        # Transaction Type Distribution
        st.subheader("Transaction Types")
        fig_types = px.pie(
            apt_transactions,
            names='Transaction_Type',
            title='Transaction Type Distribution'
        )
        st.plotly_chart(fig_types)

        # Top Categories
        st.subheader("Top Categories")
        category_stats = apt_transactions.groupby('Category')['Weight'].sum().sort_values(ascending=False)
        fig_categories = px.bar(
            category_stats.reset_index(),
            x='Category',
            y='Weight',
            title='Weight Reduced by Category'
        )
        st.plotly_chart(fig_categories)

    else:
        st.info("No transactions found for this apartment")

def main():
    st.title("CartSmall - Waste Reduction Marketplace")

    # Load data
    users_df, transactions_df = load_actual_data()

    if users_df is None or transactions_df is None:
        st.error("Failed to load data")
        return

    # Add debug info
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Users loaded: {len(users_df) if users_df is not None else 0}")
    st.sidebar.write(f"Transactions loaded: {len(transactions_df) if transactions_df is not None else 0}")

    # User Selection
    st.sidebar.subheader("User Selection")
    if not users_df.empty:
        selected_user = st.sidebar.selectbox(
            "Select User",
            options=users_df['user_id'].tolist(),
            format_func=lambda x: users_df[users_df['user_id'] == x]['full_name'].iloc[0]
                                if not users_df[users_df['user_id'] == x].empty else "Unknown"
        )

        if selected_user:
            user_data = users_df[users_df['user_id'] == selected_user]
            if not user_data.empty:
                user_apartment = user_data['apartment'].iloc[0]

                # Main tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Dashboard",
                    "Apartment Overview",
                    "County Overview",
                    "Predictions"
                ])

                try:
                    with tab1:
                        left_col, right_col = st.columns([3, 2])
                        with left_col:
                            show_transaction_form()
                        with right_col:
                            show_dashboard(selected_user, users_df, transactions_df)

                    with tab2:
                        show_apartment_dashboard(user_apartment, transactions_df)

                    with tab3:
                        show_county_dashboard(transactions_df)

                    with tab4:
                        if transactions_df is not None:
                            show_predictions_dashboard(transactions_df)
                        else:
                            st.error("Transaction data not available for predictions")

                except Exception as e:
                    st.error(f"Error in tab rendering: {str(e)}")
                    st.write("Error location:", e.__traceback__.tb_frame.f_code.co_name)

            else:
                st.warning("No data found for selected user")
    else:
        st.error("No user data available")

if __name__ == "__main__":
    main()
