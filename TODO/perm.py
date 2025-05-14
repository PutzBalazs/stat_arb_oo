import pandas as pd
import numpy as np
from lightweight_charts import Chart
from stat_arb.storage import JoblibDataStorage
from stat_arb.dex import OneInchDex

def test_clean_solution():
    """Test with the clean solution - chronological order but with volume=0"""
    data_storage = JoblibDataStorage(base_path="data/")
    tokens = data_storage.read("dex", "oneinch_tokens")
    token = tokens[0]
    
    print(f"\nClean Solution Test: {token.symbol} in chronological order")
    
    # Get the original data and sort in chronological order (IMPORTANT FIX)
    ohlc_data = token.ohlc_data.copy().sort_values('timestamp', ascending=True).reset_index(drop=True)
    
    # Format date properly
    ohlc_data['time'] = pd.to_datetime(ohlc_data['timestamp']).dt.strftime('%Y-%m-%d')
    
    # Select required columns and set volume to 0 to test if zero volume is an issue
    ohlc_data = ohlc_data[['time', 'open', 'high', 'low', 'close']]
    ohlc_data['volume'] = 0
    
    print("Chronologically ordered data with volume=0:")
    print(ohlc_data.head())
    
    # Create chart with the clean data
    chart = Chart(toolbox=True)
    chart.set(ohlc_data)
    chart.show(block=True)

def test_enhanced_chart():
    """Create an enhanced chart with custom colors and TradingView-like features"""
    data_storage = JoblibDataStorage(base_path="data/")
    tokens = data_storage.read("dex", "oneinch_tokens")
    token = tokens[0]
    ohlc_data = token.ohlc_data.copy().sort_values('timestamp', ascending=True).reset_index(drop=True)
    ohlc_data['time'] = pd.to_datetime(ohlc_data['timestamp']).dt.strftime('%Y-%m-%d')
    ohlc_data = ohlc_data[['time', 'open', 'high', 'low', 'close']]
    ohlc_data['volume'] = 0
    chart = Chart(toolbox=True)
    chart.set(ohlc_data)
    chart.layout(background_color='#FFFACD',  text_color='#000000')
    chart.grid(vert_enabled=False, horz_enabled=False)
    chart.show(block=True)

if __name__ == "__main__":
    print("Creating TradingView-like chart with custom styling...")
    test_enhanced_chart() 