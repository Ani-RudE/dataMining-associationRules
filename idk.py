import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp

df=pd.read_csv("groceriesDataset.csv")
df.head(10)





#EDA
df.info() #Summary of dataset

df.shape

df.columns

df.count()

df.dtypes

df.nunique()





#Data Cleaning and Preprocessing
df.isnull().sum()

df['date']=pd.to_datetime(df["Date"]) #Type conversion of "Date" col to `datetime` and storing it in a new col named "date"
df.info()





#Data Visualization
itemDstbn=df.groupby(by='itemDescription').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False).head(10)

bars=itemDstbn["itemDescription"]
height=itemDstbn["Frequency"]
xPos=np.arange(len(bars))

mp.figure(figsize=(16,5))
# mp.bar(xPos, height, color=(0.2, 0.3, 0.5, 0.5))
mp.bar(xPos, height, color=(0.69, 0, 1, 0.75))
mp.title("Top 10 Sold Items")
mp.xlabel("Item Names")
mp.ylabel("Quantity Sold")
mp.xticks(xPos, bars)
mp.show()

df_date = df.set_index(['Date'])
df_date

import pandas as pd

# Example DataFrame
data = df

df_date = pd.DataFrame(data)
df_date["date"] = pd.to_datetime(df_date["date"])  # Convert the date column to a DateTime format
df_date.set_index("date", inplace=True)  # Set the DateTime column as the index

# Now you can use resample
df_date.resample("M")["itemDescription"].count().plot(figsize=(20, 8), grid=True, title='Number of items sold by month').set(xlabel="Date", ylabel="Number of items sold")





#Apriori Association Rules
cust_level = df[["Member_number", "itemDescription"]].sort_values(by = "Member_number", ascending = False)
cust_level["itemDescription"]=cust_level["itemDescription"].str.strip()
cust_level

transactions = [a[1]['itemDescription'].tolist() for a in list(cust_level.groupby(["Member_number"]))]

# !pip install apyori

from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.002,min_confidence=0.05,min_left=3,min_length=2)


results = list(rules)

results

def inspect(results):
  lhs=[tuple(result[2][0][0])[0] for result in results]
  rhs=[tuple(result[2][0][1])[0] for result in results]
  supports=[result[1] for result in results]
  confidences=[result[2][0][2] for result in results]
  lifts=[result[2][0][3] for result in results]
  return list(zip(lhs,rhs,supports,confidences,lifts))





#Recommendation System
def get_recommendations(selected_product, results, min_support, min_confidence, min_lift, max_recommendations=3):
    recommendations = []

    for rule in results:
        items = list(rule.items)
        if len(items) > 1:
            base_item = items[0]
            add_item = items[1]
            support = rule.support
            confidence = rule.ordered_statistics[0].confidence
            lift = rule.ordered_statistics[0].lift

            if base_item == selected_product and support > min_support and confidence > min_confidence and lift >= min_lift:
                recommendations.append((add_item, support, confidence, lift))

    # Sort recommendations by lift
    recommendations.sort(key=lambda x: x[3], reverse=True)

    # Return at most max_recommendations recommendations
    return recommendations[:max_recommendations]

selected_product = input("Enter the product: ")
min_lift_threshold = 1.0
recommendations = get_recommendations(selected_product, results, min_support=0.002, min_confidence=0.05, min_lift=min_lift_threshold, max_recommendations=3)

if recommendations:
    print(f"Top 3 recommendations for '{selected_product}':")
    for add_item, support, confidence, lift in recommendations:
        print(f"Buy '{add_item}' (Support: {support}, Confidence: {confidence}, Lift: {lift})")
else:
    print(f"No recommendations found for '{selected_product}' with lift >= {min_lift_threshold}.")





#Time Series Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming 'Frequency' column is the numerical column representing quantity sold
ts_data = itemDstbn['Frequency']

# Convert the index to datetime
ts_data.index = pd.to_datetime(ts_data.index)

# Step 3: Decompose Time Series Components
chunk_size = 100  # Adjust the chunk size as needed
num_chunks = len(ts_data) // chunk_size + 1

plt.figure(figsize=(15, 10))

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size
    chunk = ts_data[start_idx:end_idx]

    decomposition = seasonal_decompose(chunk, period=5)  # Use a smaller 'period'

    plt.subplot(num_chunks, 4, i * 4 + 1)
    plt.plot(chunk, label='Original (Chunk {})'.format(i + 1))
    plt.title('Original Time Series')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend(loc='upper left')

    plt.subplot(num_chunks, 4, i * 4 + 2)
    plt.plot(decomposition.trend, label='Trend')
    plt.title('Trend')
    plt.xlabel('Date')
    plt.ylabel('Trend')
    plt.legend(loc='upper left')

    plt.subplot(num_chunks, 4, i * 4 + 3)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.title('Seasonality')
    plt.xlabel('Date')
    plt.ylabel('Seasonality')
    plt.legend(loc='upper left')

    plt.subplot(num_chunks, 4, i * 4 + 4)
    plt.plot(decomposition.resid, label='Residuals')
    plt.title('Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Assuming 'Frequency' column is the numerical column representing quantity sold
ts_data = itemDstbn['Frequency']

# Convert the index to datetime
ts_data.index = pd.to_datetime(ts_data.index)

# Step 4: Choose a Time Series Forecasting Model (ARIMA)
# Try different ARIMA(p, d, q) parameters
p, d, q = 3, 1, 1

# Step 5: Train the Time Series Model
train_size = int(len(ts_data) * 0.8)
train, test = ts_data[:train_size], ts_data[train_size:]

# Provide the frequency information to the ARIMA model
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Step 6: Evaluate the Model
predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plotting the results
plt.figure(figsize=(15, 10))

# Plot Training Data
plt.subplot(3, 1, 1)
plt.plot(train, label='Training Data')
plt.title('Training Data')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()

# Plot Test Data
plt.subplot(3, 1, 2)
plt.plot(test, label='Test Data')
plt.title('Test Data')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()

# Plot Predictions
plt.subplot(3, 1, 3)
plt.plot(predictions, label='Predictions', linestyle='dashed')
plt.title('Model Predictions')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()

plt.tight_layout()
plt.show()