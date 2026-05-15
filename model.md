# Bitcoin Low Price Prediction Model

The development of the Bitcoin low price prediction model followed a rigorous statistical methodology designed to prioritize "trader friendliness"—meaning the models are easy to implement and interpret while maintaining high accuracy [1] [2].

## Model Development Process

The development of the predictive models was executed through the following sequential steps:

1. **Data Collection and Categorization:** The researcher identified 23 independent variables categorized as endogenous (internal to the Bitcoin network, such as hash rate and mining difficulty) or exogenous (external factors like U.S. GDP, Gold prices, and Google Trends) [3] [4].
2. **Log Transformation:** Both independent and dependent variables (high and low prices) were log-transformed. This "Log-Log" transformation was applied to help the models better detect trends within the data that might be missed using standard dollar units [5] [6] [7].
3. **Variable Selection via Regression:** Each of the 23 variables was individually regressed against the low price. Variables were evaluated based on their Mean Absolute Percent Error (MAPE). Following the Lewis (1982) benchmark, only variables with a MAPE under 10% were considered [8] [9].
4. **Simplification:** To align with the goal of simplicity, **Bitcoin’s open price** was selected as the sole independent variable for the final ARDL models because it was identified as the most significant predictor with the lowest MAPE [10].
5. **Stationarity and Lag Optimization:** An Augmented Dickey Fuller (ADF) test was conducted to ensure data stationarity. The researcher then used correlograms and partial correlation functions to determine the optimal lag structures for the Autoregressive Distributed Lags (ARDL) models [11].
6. **Regime-Dependent Training:** Models were developed using regime-dependent analysis to account for different market structures. This resulted in three types of ARDL models: General, Bull Market (trained on the 2020 run), and Bear Market (trained on the 2017 run) [12] [13] [14].

---

## Optimal Model for Low Price Prediction

After comparing approximately 24 statistical models across daily, weekly, and monthly frequencies, the study identified the **ARDL Bull Monthly** model as the optimal choice for predicting Bitcoin's low price [15] [16].

- **Frequency:** Monthly [17] [18].
- **Independent Variable:** Bitcoin Open Price [19] [20].
- **Lag Structure:** (5,2) [21].
- **Training Performance:** It achieved a training set MAPE of **0.130314%**, which was the lowest among all low-price models [22] [23].

---

## Robustness and Results

The ARDL Bull Monthly model was subjected to seven robustness tests using out-of-sample data from 2011 to 2024 to ensure it could generalize to various market conditions [24] [25].

| Test Sample                       | MAPE      |
| :-------------------------------- | :-------- |
| 2012 Bull Run                     | 6.052339% |
| 2013 Bear Run                     | 4.591585% |
| 2016 Bull Run                     | 1.634181% |
| 2017 Bear Run                     | 2.024038% |
| Full Historical Sample            | 6.734580% |
| 2021 Bear Run (Forward Test)      | 1.469782% |
| 2023 Mini Bull Run (Forward Test) | 0.472748% |

The model maintained an average MAPE of **3.28%** across these tests, significantly below the 10% threshold for high forecasting accuracy [26] [27] [28]. A trading demonstration showed that using this model's predicted low price as an entry point could potentially generate profits ranging from $7.00 to $2,099.86 per 1 BTC, depending on the market regime [29].
