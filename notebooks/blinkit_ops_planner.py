import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score

# Load data (replace with your path)
df = pd.read_csv(r'C:\Users\RADHA KRISHNA\Downloads\data analysis project\data\blinkit_cleaned_data.csv', 
                 parse_dates=['order_date'])

# Feature Engineering
def preprocess_data(df):
    # Convert to daily aggregates
    daily = df.groupby(pd.Grouper(key='order_date', freq='D')).agg(
        orders=('order_id', 'count'),
        avg_order_value=('order_total', 'mean'),
        promo_orders=('order_total', lambda x: sum(x < 0.9*df['mrp'].mean()))  # 10% discount threshold
    ).reset_index()
    
    # Add critical features
    daily['day_of_week'] = daily['order_date'].dt.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5,6]).astype(int)
    daily['month'] = daily['order_date'].dt.month
    daily['rolling_7'] = daily['orders'].rolling(7).mean().ffill()
    
    return daily

daily_data = preprocess_data(df)