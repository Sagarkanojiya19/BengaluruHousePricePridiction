# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru using linear regression and real-world housing data. This end-to-end project includes data cleaning, feature engineering, model training, and deployment as a web app.

### Software And Tools Requirements

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [Heroku Account](https://heroku.com)
4. [Git CLI](https://git-scm.com/downloads)


## 📊 Dataset

- **Source**: [Kaggle - Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- **Features include**:
  - Location
  - Total Square Footage
  - Number of Bedrooms (BHK)
  - Number of Bathrooms
  - Price (Target Variable)

## 🧹 Data Cleaning & Preprocessing

- Removed unnecessary columns (`area_type`, `society`, `availability`, etc.)
- Handled missing values
- Handled inconsistent formats (e.g., square footage ranges like `2100 - 2850`)
- Reduced dimensionality of categorical features (e.g., grouped rare locations under "other")
- Removed outliers (e.g., 2 BHK houses with less than 600 sqft)

## 📈 Model

- **Algorithm Used**: RandomForestRegressor
- **Techniques**:
  - One-Hot Encoding for categorical variables
  - Train-Test Split
  - GridSearchCV for best parameter selection
- **Evaluation Metrics**:
  - R² Score
  - Residual Analysis

## 🌐 Web App

An interactive web app built to let users input property details and get instant price predictions.

- **Frontend**: Streamlit 
- **Deployment**: [Streamlit Share](https://share.streamlit.io/)

🔗 **Live Demo**: [BengaluruHousePricePrediction](https://bengaluruhousepricepridictiongi-bcugrjytde2jzwinufc4gn.streamlit.app/)
💻 **Code**: [BengaluruHousePricePridiction](https://github.com/Sagarkanojiya19/BengaluruHousePricePridiction)

## 🛠️ Tech Stack

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 📌 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Sagarkanojiya19/BengaluruHousePricePridiction.git
cd BengaluruHousePricePridiction

# Install dependencies
pip install -r requirements.txt

# Run the notebook (for EDA and model building)
jupyter notebook Bengaluru_House_Price_Prediction.ipynb

# OR launch the web app
streamlit run app.py  # if using Streamlit


