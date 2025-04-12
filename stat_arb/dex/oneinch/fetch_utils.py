import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


# TODO: require API key
load_dotenv()
API_KEY = os.getenv('API_KEY')
#print(f"API_KEY: {API_KEY}")

def get_all_tokens(chain_id=1):
    print(API_KEY)
    apiUrl = f"https://api.1inch.dev/token/v1.2/{chain_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(apiUrl, headers=headers)
    return response.json()

def get_ohlc_data(token0_address, token1_address, chain_id, granularity, limit):
    """Get OHLC price data for a token pair"""
    method = "get"
    apiUrl = "https://api.1inch.dev/portfolio/integrations/prices/v1/time_range/cross_prices"
    requestOptions = {
        "headers": {
            "Authorization": f"Bearer {API_KEY}"
        },
        "body": "",
        "params": {
            "token0_address": token0_address,
            "token1_address": token1_address,
            "chain_id": chain_id,
            "granularity": granularity,
            "limit": limit
        }
    }

    # Prepare request components
    headers = requestOptions.get("headers", {})
    params = requestOptions.get("params", {})

    response = requests.get(apiUrl, headers=headers, params=params)

    return response

def extract_basic_token_info(all_tokens, max_tokens=None):
    """Extract only the fields we need from the token data"""
    if max_tokens is None:
        max_tokens = len(all_tokens)
    basic_token_info = {}
    counter = 0

    for address, info in all_tokens.items():
        if counter >= max_tokens:
            break

        basic_token_info[address] = {
            "basic_info": {
                "name": info.get("name"),
                "symbol": info.get("symbol"),
                "decimals": info.get("decimals"),
                "logo": info.get("logoURI")
            }
        }
        counter += 1

    return basic_token_info


def fetch_ohlc_for_tokens(token_data, chain_id, granularity="day", limit=30, api_wait=1):
    """Fetch OHLC data for each token against ETH"""
    eth_address = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"  # ETH address
    tokens_with_data = {}
    failed_tokens = []

    for address, data in token_data.items():
        time.sleep(api_wait)  # Rate limit to 1 request per second
        print(f"Fetching OHLC data for {data['basic_info']['symbol']}...")

        ohlc_response = get_ohlc_data(address, eth_address, chain_id, granularity, limit)

        if ohlc_response.status_code == 200:
            try:
                response_data = ohlc_response.json()

                if isinstance(response_data, list) and len(response_data) > 0:
                    df = pd.DataFrame(response_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

                    # Only include if we have the full requested number of data points
                    if len(df) >= limit:
                        tokens_with_data[address] = data.copy()
                        tokens_with_data[address]["ohlc_data"] = df
                        print(f"  Got {len(df)} data points (minimum required: {limit})")
                    else:
                        print(f"  Skipping {data['basic_info']['symbol']} - only {len(df)} data points (minimum required: {limit})")
                        failed_tokens.append(address)
                elif isinstance(response_data, dict) and "error" in response_data:
                    print(f"  API Error: {response_data['error']}")
                    failed_tokens.append(address)
                else:
                    print(f"  Unexpected response format")
                    failed_tokens.append(address)
            except Exception as e:
                print(f"  Error parsing response: {str(e)}")
                failed_tokens.append(address)
        else:
            print(f"  Error: {ohlc_response.status_code}")
            failed_tokens.append(address)

    if failed_tokens:
        print(f"\nFailed to fetch sufficient data for {len(failed_tokens)} tokens:")
        for address in failed_tokens:
            print(f"  - {token_data[address]['basic_info']['symbol']}")

    print(f"\nSuccessfully fetched data for {len(tokens_with_data)} tokens with sufficient data points")
    return tokens_with_data


