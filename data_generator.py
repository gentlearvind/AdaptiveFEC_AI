import csv
import random

class DataGenerator:
    def __init__(self, num_data_points=100, file_name="impairments.csv"):
        self.num_data_points = num_data_points
        self.file_name = file_name
        self.parameters = {
            "span_length_km": (50, 1000),  # Span length in km
            "chromatic_dispersion_ps_nm_km": (16, 20),  # Chromatic dispersion in ps/nm/km
            "polarization_mode_dispersion_ps_sqrt_km": (0.1, 1),  # PMD in ps/sqrt(km)
            "attenuation_dB_km": (0.2, 0.5),  # Attenuation in dB/km
            "nonlinear_coefficient_W_1_km_1": (1.1e-3, 2.6e-3),  # Nonlinear coefficient in W^-1 km^-1
            "optical_signal_to_noise_ratio_dB": (15, 30),  # OSNR in dB
            "SPM": (0.01, 0.05),  # Self-Phase Modulation
            "XPM": (0.01, 0.05),  # Cross-Phase Modulation
            "FWM": (0.01, 0.05),  # Four-Wave Mixing
            "insertion_loss_dB": (0.5, 2),  # Insertion Loss in dB
            "modulation_format": ["QPSK", "16QAM", "64QAM"],  # Modulation formats
            "FEC_algorithm": ["RS" + str(i) for i in range(239, 127, -1)] + ["LDPC", "Turbo"]  # FEC algorithms with RS codes from RS239 to RS128
        }

    def generate_data(self):
        data = []
        for _ in range(self.num_data_points):
            data_point = {
                "span_length_km": random.uniform(*self.parameters["span_length_km"]),
                "chromatic_dispersion_ps_nm_km": random.uniform(*self.parameters["chromatic_dispersion_ps_nm_km"]),
                "polarization_mode_dispersion_ps_sqrt_km": random.uniform(*self.parameters["polarization_mode_dispersion_ps_sqrt_km"]),
                "attenuation_dB_km": random.uniform(*self.parameters["attenuation_dB_km"]),
                "nonlinear_coefficient_W_1_km_1": random.uniform(*self.parameters["nonlinear_coefficient_W_1_km_1"]),
                "optical_signal_to_noise_ratio_dB": random.uniform(*self.parameters["optical_signal_to_noise_ratio_dB"]),
                "SPM": random.uniform(*self.parameters["SPM"]),
                "XPM": random.uniform(*self.parameters["XPM"]),
                "FWM": random.uniform(*self.parameters["FWM"]),
                "insertion_loss_dB": random.uniform(*self.parameters["insertion_loss_dB"]),
                "modulation_format": random.choice(self.parameters["modulation_format"]),
                "FEC_algorithm": random.choice(self.parameters["FEC_algorithm"])
            }
            data.append(data_point)

        with open(self.file_name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"Data has been generated and written to {self.file_name}")