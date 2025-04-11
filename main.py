
from stat_arb.storage import JolibDataStorage
from stat_arb.dex import OneInchDex

def main():
    # Initialize JolibDataStorage (implements DataStorage)
    data_storage = JolibDataStorage(base_path="data/step_1")
    dex = OneInchDex(chain_id=42161)
    dex.asd()

if __name__ == "__main__":
    main()
