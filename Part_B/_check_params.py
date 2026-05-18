import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import v2_SystemCharacteristics as sc
raw = sc.get_fixed_data()
print("eta_occ:", raw['humidity_occupancy_coeff'])
print("eta_vent:", raw['humidity_vent_coeff'])
print("H_high:", raw['humidity_threshold'])
print()
# Simulate humidity over 10 steps without ventilation, occ1=30, occ2=20
H = 40.0
eta_occ = raw['humidity_occupancy_coeff']
eta_vent = raw['humidity_vent_coeff']
H_high = raw['humidity_threshold']
print("Humidity evolution (no ventilation, occ1=30, occ2=20):")
for t in range(10):
    H += eta_occ * (30 + 20)
    print(f"  t={t}: H={H:.2f}  {'FORCED VENT!' if H >= H_high else ''}")