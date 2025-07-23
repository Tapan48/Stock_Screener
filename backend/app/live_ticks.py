
import logging
from kiteconnect import KiteTicker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

# Load API credentials from environment
api_key_tapan = os.getenv("API_KEY_TAPAN")
api_secret_tapan = os.getenv("API_SECRET_TAPAN")
access_token_tapan = os.getenv("ACCESS_TOKEN_TAPAN")



# Initialise
kws = KiteTicker(api_key_tapan, access_token_tapan)

def on_ticks(ws, ticks):
    # Callback to receive ticks.
    # logging.debug("Ticks: {}" .format(ticks))
    # print(ticks)
    # Convert ticks to a format suitable for visualization
    for tick in ticks:
        print("Tick data:")
        print(tick)


def on_connect(ws, response):
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    ws.subscribe([738561])

    # Set RELIANCE to tick in `full` mode.
    ws.set_mode(ws.MODE_FULL, [738561])

def on_close(ws, code, reason):
    # On connection close stop the main loop
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()

# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
# Use threaded=True to avoid reactor issues
kws.connect()
