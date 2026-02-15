import numpy as np

# Test NPV calculation
first_year = 1915710467.41  # kWh
discount_rate = 0.0749997
degradation = 0.005

# Method 1: Excel way (from compare_lcoe_components.py that worked)
energy_crore_kwh = []
for year in range(25):
    degraded = (first_year / 1e7) * ((1 - degradation) ** year)
    energy_crore_kwh.append(degraded)

npv_excel = sum(energy_crore_kwh[year] / ((1 + discount_rate) ** (year + 1)) for year in range(25))
print(f"Excel method NPV: {npv_excel:.2f} Crore kWh")
print(f"Expected: 2048.22 Crore kWh")
print(f"Match: {abs(npv_excel - 2048.22) < 0.01}")

# Method 2: calculate_lcoe way
energy_kwh = np.full(25, first_year)
npv_lcoe = 0.0
for year in range(25):
    n = year + 1
    degraded_energy = energy_kwh[year] * ((1 - degradation) ** (n - 1))
    npv_lcoe += degraded_energy / ((1 + discount_rate) ** n)

print(f"\ncalculate_lcoe method NPV: {npv_lcoe:.2f} kWh = {npv_lcoe/1e7:.2f} Crore kWh")
print(f"Expected: 2048.22 Crore kWh")
print(f"Match: {abs(npv_lcoe/1e7 - 2048.22) < 0.01}")
