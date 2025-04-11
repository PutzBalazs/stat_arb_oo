from . import fetch_utils
class OneInchDex:
    def __init__(self, chain_id=1, granularity="1d", limit=30, api_wait=1, max_tokens=100):
        self.chain_id = chain_id
        self.granularity = granularity
        self.limit = limit
        self.api_wait = api_wait
        self.max_tokens = max_tokens

    def asd(self):
        print("asd")
        all_tokens = fetch_utils.get_all_tokens(self.chain_id)
        print(f"Found {len(all_tokens)} tokens")
        print("Extracting basic token info...")
        
        
    def fetch_and_prepare_data(self):
        print("Fetching tokens...")
        all_tokens = fetch_utils.get_all_tokens(self.chain_id)
        print(f"Found {len(all_tokens)} tokens")

        print("Extracting basic token info...")
        token_data = fetch_utils.extract_basic_token_info(all_tokens, self.max_tokens)
        print(f"Selected {len(token_data)} tokens for analysis")

        print("Fetching OHLC data...")
        token_data = fetch_utils.fetch_ohlc_for_tokens(
            token_data, self.chain_id, self.granularity, self.limit, self.api_wait
        )
        print(f"Got price data for {len(token_data)} tokens")

        return token_data
