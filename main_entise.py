
import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Columns as Cols
from entise.constants import Types
from examples.utils import load_input, run_simulation

def main(print_summary: bool = False, analysis: bool = False, save_figures=False) -> None:
    objects, data = load_input()
    summary, df = run_simulation(objects, data, workers=1)
    if print_summary:
        print("Summary:")
        print(summary.to_string())


if __name__ == "__main__":
    main(print_summary=True, analysis=True, save_figures=False)
