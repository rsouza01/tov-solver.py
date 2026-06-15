import pandas as pd
import os

class EoSExporter:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_to_csv(self, eps, p, filename="eos_data.csv"):
        """
        Exports EoS data to a CSV file.
        Format: energy_density, pressure
        """
        df = pd.DataFrame({'energy_density': eps, 'pressure': p})
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, sep=' ') # Space-separated for easy Gnuplot reading
        print(f"Data exported to {filepath}")