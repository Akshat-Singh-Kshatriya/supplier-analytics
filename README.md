
# Critical Mineral Pricing & Supplier Analytics

## Project Overview
A supply chain analytics project focused on forecasting critical mineral prices, evaluating procurement strategies and analysing supplier performance to support cost-effective sourcing decisions for lithium-ion battery manufacturing.

The project combines machine learning, big data processing and supplier scorecard analysis to evaluate procurement costs and supply chain efficiency.

## Objectives
- Forecast critical mineral prices over a three-month horizon.
- Evaluate forecast-driven procurement strategies.
- Analyse supplier delivery and rejection performance.
- Compare supplier quotations against adjusted landed costs.
- Identify opportunities for procurement cost optimisation.

## Tech Stack
- **Programming:** Python
- **Big Data Processing:** PySpark, Spark SQL
- **Machine Learning:** Scikit-learn, Gradient Boosting
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Power BI, Matplotlib, Seaborn

## Dataset

The project uses two illustrative datasets:

### 1. Commodity Price Dataset
- Historical mineral prices.
- Monthly price movements.
- Lagged price features.
- Rolling averages and market trends.

### 2. Supplier Performance Dataset
- 2,000 supplier delivery records.
- Eight supplier profiles.
- Quoted procurement prices.
- Rejection rates.
- Delivery performance.
- Adjusted landed costs.

## Methodology

### 1. Commodity Price Forecasting
- Cleaned and transformed historical commodity price data.
- Engineered lagged and rolling statistical features.
- Developed a Gradient Boosting forecasting model.
- Applied walk-forward validation across 32 months.
- Compared forecast performance against a naive baseline using MAPE.

### 2. Procurement Strategy Analysis
- Developed a forecast-based purchasing strategy.
- Evaluated forward purchasing decisions when predicted price increases exceeded 3%.
- Compared procurement costs against spot-market purchasing.
- Estimated potential savings under simulated market conditions.

### 3. Supplier Performance Analytics
- Processed 2,000 supplier delivery records using PySpark and Spark SQL.
- Evaluated supplier quotations, rejection rates and delivery performance.
- Calculated adjusted landed costs using illustrative quality and delivery penalties.
- Developed supplier scorecards to compare procurement performance.
- Identified suppliers offering the lowest adjusted procurement costs.

## Key Results

| Performance Metric | Result |
|---|---:|
| Forecast horizon | 3 months |
| Gradient Boosting MAPE | 8.8% |
| Naive baseline MAPE | 14.2% |
| Forecast error improvement | 38% |
| Potential procurement savings | 6.2% |
| Supplier delivery records | 2,000 |
| Suppliers evaluated | 8 |
| Best adjusted supplier cost | $15,911/tonne |


## Business Impact
The project demonstrates how predictive analytics and supplier performance evaluation can improve procurement planning, support strategic sourcing and reduce exposure to critical mineral price fluctuations.

## Assumptions and Limitations
- Commodity price data and supplier records are illustrative.
- Forecast performance is based on walk-forward validation over the available dataset.
- Procurement savings are simulated and do not represent realised purchasing savings.
- Supplier delivery penalties are based on illustrative assumptions.
