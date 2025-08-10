# vizzy/rpi/__main__.py
from .config import parse_args
from .server import main as run

if __name__ == "__main__":
    args = parse_args()
    run(debug=bool(args.debug))
