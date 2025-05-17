from game import run_singleplayer # , run_multiplayer
from config import config
import sys

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "multiplayer":
        print("Multiplayer mode is not implemented yet.")
        # run_multiplayer(**config)
    else:
        print("Singleplayer mode")
        run_singleplayer(**config)