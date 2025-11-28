



## Stock Market Price Estimation

Stock market price estimation is a crucial aspect of financial analysis that impacts investors, companies, and analysts alike. The goal is to achieve accurate and reliable results to guide decision-making and minimize risks.

Inaccurate estimations can lead to:

- **Investor Losses:** Misleading predictions may cause poor investment decisions.
- **Market Panic:** Overly pessimistic forecasts can trigger unnecessary sell-offs.
- **Reputational Risk:** Financial institutions or analysts may lose credibility.

My aim is to analyze stock market price data for Google, Meta, Apple, and NVIDIA between October 30, 2021, and October 30, 2024, and provide an estimator while deploying my model both locally and to the cloud.

My project focuses on predicting the stock market's closing price for the 61st day based on the preceding 60 days of data. This provides an estimated future stock price, which can assist individuals in evaluating investment opportunities. Users can compare the predicted prices of companies such as Google, Meta, Apple, or NVIDIA to make more informed decisions about where to invest.





## [Data Collection]

I used the [ ```yfinance```](https://github.com/ranaroussi/yfinance/tree/main?tab=readme-ov-file) library to download stock market price data for four companies—Google, Meta, Apple, and NVIDIA—from October 30, 2021, to October 30, 2024. The yfinance library is a Python tool for accessing historical market data, financial metrics, and company information from [Yahoo!Ⓡ finance](https://finance.yahoo.com/).

You can download the data in two steps as outlined below:


1) Specify the companies.

```tickers = ['GOOGL', 'META', 'AAPL', 'NVDA']```

2) Decide on the start and end dates, then download the data.

```stock_data = yf.download(tickers, start="2021-10-30", end="2024-10-30", group_by="ticker")```



## Stock Price Dataset Description

This dataset contains daily stock price information, which is useful for analyzing stock performance, predicting trends, or conducting financial research.

| **Column Name** | **Description**                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------|
| **Date**         | The date of the record in the format `YYYY-MM-DD`.                                                   |
| **Open**         | The stock's opening price at the start of the trading day.                                           |
| **High**         | The highest price the stock reached during the trading day.                                          |
| **Low**          | The lowest price the stock fell to during the trading day.                                           |
| **Close**        | The stock's closing price at the end of the trading day.                                             |
| **Adj Close**    | The closing price adjusted for corporate actions like stock splits, dividends, and rights offerings. |
| **Volume**       | The total number of shares traded during the trading day. This reflects market activity and liquidity.|

In my project, I chose to analyze the __Close__ value from the stock price prediction dataset because it is widely regarded as one of the most reliable indicators for stock market analysis. The close value represents the final price at which a stock is traded at the end of the trading session, making it a critical metric for evaluating a stock's performance over time.

Unlike the open, low, or high values, which reflect specific moments or ranges during the trading day, the close value encapsulates the market's sentiment and activity for the entire trading session. It is often used by investors and analysts as a benchmark for decision-making, as it provides a clearer snapshot of how the stock performed on a given day.

## [EDA (Exploratory Data Analysis)]

### Comparison of Moving Averages:



<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Google_stock_price_MA.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Meta_stock_price_MA.png" width="800" height="400" />
</p>

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Apple_stock_price_MA.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Nvidia_stock_price_MA.png" width="800" height="400" />
</p>



- Google and NVIDIA show the strongest trends with consistent golden crosses, indicating robust upward momentum.
Meta also demonstrates a clear upward trend but with slightly more volatility.

- Apple exhibits more fluctuations in short-term trends, with the 20-day MA crossing below the 50-day MA occasionally, reflecting periods of weaker performance.

__Actionable Insights:__

- Google and NVIDIA are in strong bullish phases, suggesting potential opportunities for growth-focused investments.
- Meta shows recovery signs, making it an interesting candidate for medium-term strategies.
- Apple’s performance might require closer monitoring of short-term fluctuations to better time buy or sell decisions.

### Time Series Analysis Comments for Google, Meta, Apple, and NVIDIA

__1. Google__

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Google_Trend.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Google_Seasonal.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Google_Residual.png" width="800" height="400" />
</p>


__Trend:__ The trend is consistently increasing, reflecting a steady growth pattern in the time series over the given period. This could indicate positive long-term growth or increased activity.
__Seasonality:__ The seasonal component exhibits moderate fluctuations, showing periodic patterns that are relatively consistent. This suggests some underlying cyclical behavior in the data.
__Residual:__ The residuals are small and do not show significant spikes, indicating that the model captures the trend and seasonality well. The minimal noise may explain why I achieved the best score for Google.

__2. Meta__

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Meta_Trend.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Meta_Seasonal.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Meta_Residual.png" width="800" height="400" />
</p>

__Trend:__ The trend is upward and fairly strong, indicating a significant increase in values over time. This suggests positive momentum in the dataset.
__Seasonality:__ Meta's seasonal component appears more volatile compared to Google, with frequent and irregular fluctuations. This suggests that the data has a more complex periodic structure.
__Residual:__ The residuals show more variability and larger deviations compared to Google, meaning that the model may not fully capture all aspects of the time series.

__3. Apple__

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Apple_Trend.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Apple_Seasonal.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/Apple_Residual.png" width="800" height="400" />
</p>



__Trend:__ The trend is upward and consistent but slightly less steep than NVIDIA and Meta. It indicates steady growth over time.
__Seasonality:__ Apple’s seasonal component shows relatively higher variability and irregular periodic patterns. This complexity in seasonality could make modeling more challenging.
__Residual:__ The residuals exhibit higher noise, suggesting that the model struggles to explain the variation in the data. This aligns with the  observation that Apple's classification results are the worst.

__4. NVIDIA__

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/NVIDIA_Trend.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/NVIDIA_Seasonal.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/NVIDIA_Residual.png" width="800" height="400" />
</p>



__Trend:__ The trend for NVIDIA shows a strong and consistent upward movement, similar to Meta but with steeper growth. This indicates rapid changes or increases in the series.
__Seasonality:__ NVIDIA's seasonal component is less volatile than Apple or Meta, showing periodic fluctuations that are smoother and more predictable.
__Residual:__ The residuals are relatively well-contained but exhibit some spikes, indicating occasional deviations from the model’s predictions.


### Removing Outliers 

<p float="left">
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/close_disrubition.png" width="800" height="400" />
  <img src="https://github.com/devparth19/Stock-Market-Prediction/blob/main/eda/close_disrubition_outliers.png" width="800" height="400" />
</p>


I conducted an outlier analysis for the four companies (Google, Meta, Apple, and Nvidia) and identified and removed outliers from the dataset. After the cleaning process, I saved the updated datasets to the following files:

- [google_stock_cleaned.csv](https://github.com/devparth19/Stock-Market-Prediction/blob/main/inputs/google_stock_cleaned.csv)

- [meta_stock_cleaned.csv](https://github.com/devparth19/Stock-Market-Prediction/blob/main/inputs/meta_stock_cleaned.csv)

- [apple_stock_cleaned.csv](https://github.com/devparth19/Stock-Market-Prediction/blob/main/inputs/apple_stock_cleaned.csv)

- [nvidia_stock_cleaned.csv](https://github.com/devparth19/Stock-Market-Prediction/blob/main/inputs/nvidia_stock_cleaned.csv)


For the remainder of the project, I used this cleaned data for all subsequent experiments and analyses to ensure improved data quality and reliable results.

## Training

### Models
#### Recurrent Neural Networks (RNNs)

RNNs are designed to process sequential data by maintaining a memory of past inputs. This makes them a natural choice for time series, where patterns over time are critical. However, they struggle with long-term dependencies due to the vanishing gradient problem. While simpler and faster to train than some advanced models, their limited capacity to capture long-range relationships can be a drawback for complex time series.

__Advantages:__

- Simple architecture that captures short-term dependencies well.
- Computationally less expensive compared to more complex models.

__Disadvantages:__

- Struggles with long-term dependencies.
- Prone to vanishing gradient issues, leading to poorer performance on long sequences.

#### Neural Networks (NNs)

Standard neural networks (e.g., feedforward networks) are less commonly used in time series because they don't inherently account for sequential information. They treat each input as independent, which may lead to loss of critical temporal patterns unless engineered features are explicitly provided.

__Advantages:__

- Simpler to implement and train for non-sequential data or aggregated features.
- May perform well when time-dependent relationships are less critical.

__Disadvantages:__

- Does not natively capture sequential dependencies in the data.
- Requires manual feature engineering to represent time-based patterns effectively.
 
#### Long Short-Term Memory Networks (LSTMs)

LSTMs extend RNNs by incorporating memory cells and gates to selectively remember or forget information. This makes them well-suited for time series with long-term dependencies. They have been widely used in applications like stock price prediction, weather forecasting, and anomaly detection.

__Advantages:__

- Handles both short-term and long-term dependencies effectively.
- Robust to vanishing gradients, enabling better learning over extended sequences.

__Disadvantages:__

- Higher computational complexity compared to RNNs.
- Requires more tuning and longer training times

#### Transformers

Transformers revolutionized natural language processing and are increasingly applied to time series. Their self-attention mechanism allows them to capture both short-term and long-term dependencies efficiently. Transformers excel in handling irregular sampling and multivariate time series, making them powerful but computationally demanding.

__Advantages:__

- Can model long-term dependencies effectively with the self-attention mechanism.
- Handles multivariate time series and irregular time steps well.

__Disadvantages:__

- Computationally intensive, especially for large datasets or high-dimensional data.
- Requires large datasets for effective training, which can be a limitation for some time series problems.

#### Ordinary Differential Equations (ODEs)

ODE-based models are a different beast altogether. Instead of treating the data as discrete points, they model continuous changes in time, making them especially useful for time series where smooth dynamics are essential (e.g., physical systems, population growth, or epidemiology). Neural ODEs combine ODEs with neural networks, offering a flexible yet interpretable framework.

__Advantages:__

- Provides a continuous perspective on time, which is valuable for smooth or physics-inspired time series.
- More interpretable in scientific contexts, connecting directly to underlying processes.

__Disadvantages:__

- Computationally intensive to solve, especially for stiff ODEs.
- Requires domain knowledge for meaningful parameterization.

### Metrics

- __Mean Absolute Error (MAE):__ MAE is the average of the absolute differences between predicted and actual values in a dataset. It measures the magnitude of the error in prediction, without considering whether the prediction is over or under the actual value. MAE is important in time series analysis because it gives a clear and simple understanding of the average magnitude of error across all data points.
  
- __Mean Absolute Percentage Error (MAPE):__ MAPE expresses the error as a percentage of the actual values. It shows how large the prediction error is relative to the actual value, giving a more intuitive sense of the model’s accuracy in percentage terms. MAPE is commonly used in time series forecasting because it normalizes the error by dividing by the actual values, which helps to understand the relative size of the error, especially when comparing models across datasets with different magnitudes. However, MAPE can be problematic when actual values are zero or very close to zero, as this causes large percentage errors.
- __Symmetric Mean Absolute Percentage Error (sMAPE):__  sMAPE is a variation of MAPE that normalizes the error symmetrically by using both predicted and actual values in the denominator. It aims to avoid some issues with MAPE, especially when actual values are near zero. sMAPE is useful when comparing models across different time series datasets with varying scales, as it treats under- and over-forecasting equally. It helps mitigate issues caused by small actual values in MAPE and gives a more balanced error metric.

### [Hyperparameter Tuning]

I performed hyperparameter tuning separately for [Google], [Meta], [Apple], and [NVIDIA]using Optuna, an efficient hyperparameter optimization framework. Optuna was chosen for its ability to perform automated and flexible optimization through Bayesian search strategies. Unlike grid or random search, Optuna dynamically adjusts its search based on previous results, helping to find optimal hyperparameters more effectively and reducing computational overhead.


During both hyperparameter tuning and training, I rescaled the time series data using Min-Max normalization. Min-Max normalization scales the data to a fixed range, typically [0, 1], which is particularly useful for time series since the values are often continuous and can vary across different magnitudes. 

### Model Training 
I decided to use the LSTM architecture for the , [Apple], and [NVIDIA] data, as it performed better during hyperparameter tuning. For [Meta], the RNN architecture yielded marginally better results compared to LSTM model.


__Results Summary__

| **Dataset** | **Mean Absolute Error (MAE)** | **Mean Absolute Percentage Error (MAPE)** | **Symmetric Mean Absolute Percentage Error (sMAPE)** |
|-------------|--------------------------------|------------------------------------------|-----------------------------------------------------|
| Google      | 0.0205                        | 2.7047                                   | 2.5505                                              |
| Meta        | 0.0304                        | 4.4251                                   | 4.6505                                              |
| Apple       | 0.1824                        | 20.1099                                  | 22.9494                                             |
| Nvidia      | 0.0436                        | 5.0240                                   | 4.4939                                              |



![](https://github.com/devparth19/Stock-Market-Prediction/blob/main/results/google_result.png)





![](https://github.com/devparth19/Stock-Market-Prediction/blob/main/results/meta_result.png)





![](https://github.com/devparth19/Stock-Market-Prediction/blob/main/results/apple_result.png)




![](https://github.com/devparth19/Stock-Market-Prediction/blob/main/results/nvidia_result.png)


 









