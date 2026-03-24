# Linewidth Data Summary

## 689 nm Spectroscopy (Single-Photon Reference)
Configuration: ND 1.3, 2x1, 2, artig at various pulse times

| Pi Time | RID | Width (kHz) |
|---------|-----|-------------|
| 22 us | 75009 | 52.9 ± 0.92 |
| 38 us | 75013 | 47.9 ± 1.62 |
| 75 us | 75014 | 45.3 ± 1.51 |
| 9.5 us | 75021 | 58.5 ± 2.02 |
| 6 us | 75022 | 66.2 ± 1.4 |
| 4 us | 75023 | 92.7 ± 3.75 |
| 3 us | 75025 | — |
| 2 us | 75029 | 146 ± 7.7 |
| 1 us | 75033/75034 | 323 ± 20.1 |

Linewidth narrows as pi time increases, reaching a minimum ~45 kHz around 75 us, consistent with Fourier-limited broadening at short pulse times.

---

## 3-Photon Spectroscopy
Lorentzian gamma (half-width) in kHz:

| Pi Time | RID | Lorentzian γ (kHz) |
|---------|-----|--------------------|
| 2 us | 74750 | 328 ± 46 |
| 4 us | 74759 | 187 ± 22.5 |
| 6 us | 74764 | 155 ± 16.4 |
| 9.5 us | 74768 | 77.2 ± 8.6 |
| 22 us | 74772 | 31.4 ± 1.8 |
| 38 us | 74776 | 21.8 ± 1.4 |
| 75 us | 74780 | 8.7 ± 0.37 |
| 240 us | 74785 | 2.64 ± 0.11 |
| 900 us | 74788 | 0.874 ± 0.043 |
| 3 ms | 74794 | 0.247 ± 0.015 |
| 5 ms | 74798 | 0.209 ± 0.014 |

Linewidth drops from ~328 kHz at 2 us to ~0.21 kHz at 5 ms, showing the three-photon transition is Fourier-limited down to the sub-kHz level at long pulse times. Apparent saturation between 3–5 ms suggests a residual broadening floor around 0.2 kHz.
