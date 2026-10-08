from pathlib import Path
import sys


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from run_sensitivity import main


if __name__ == "__main__":
    raise SystemExit(main())