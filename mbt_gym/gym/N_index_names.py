CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = None
ASSET_PRICE_INDEX = None

BID_INDEX = 0
ASK_INDEX = 1

def get_index_map(num_assets: int) -> dict:
    return {
        "CASH_INDEX": 0,
        "INVENTORY_INDEX": 1,
        "TIME_INDEX": 1 + num_assets,
        "ASSET_PRICE_INDEX": 2 + num_assets,  # assuming mid-price per asset
    }
