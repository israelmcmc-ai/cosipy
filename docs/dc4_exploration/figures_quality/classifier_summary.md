
### Likelihood-ratio classifier removing 90% of the background events in every energy bin (S/√B change, window re-optimised)

| selection | 200–342 keV | 342–585 keV | 585–1000 keV | 1000–1710 keV | 1710–2924 keV | 2924–5000 keV | all bins |
|---|---|---|---|---|---|---|---|
| trained and evaluated on all signal files | -54.2 | -28.5 | -19.4 | -23.6 | -24.8 | -24.7 | -36.8 ± 0.5 |
| trained on 5 files, evaluated on the other 5 | -55.2 | -28.7 | -20.1 | -24.7 | -25.1 | -24.5 | -37.5 |
| best rectangular cut (D ≥ 1 cm, SQ ≥ 3, top layer; overall 90%) | -59.2 | -32.1 | -24.4 | -27.3 | -30.4 | -33.0 | -41.1 |

### Result of the classifier cut per bin

| bin | bkg removed [%] | signal kept [%] | optimal w [deg] |
|---|---|---|---|
| 200–342 keV | 90.5 | 10.4 | 6 |
| 342–585 keV | 90.4 | 12.8 | 3.5 |
| 585–1000 keV | 90.0 | 13.8 | 2.5 |
| 1000–1710 keV | 90.0 | 12.8 | 1.5 |
| 1710–2924 keV | 90.1 | 12.3 | 1.5 |
| 2924–5000 keV | 90.1 | 13.8 | 4 |

### Composition of the events kept by the classifier (shares in %)

| variable | value | all signal | kept signal | kept background |
|---|---|---|---|---|
| first-hit layer | bottom | 13 | 0 | 0 |
| first-hit layer | 2 | 17 | 0 | 0 |
| first-hit layer | 3 | 26 | 10 | 12 |
| first-hit layer | top | 45 | 90 | 88 |
| second-hit layer | bottom | 13 | 5 | 5 |
| second-hit layer | 2 | 20 | 21 | 21 |
| second-hit layer | 3 | 29 | 40 | 41 |
| second-hit layer | top | 38 | 34 | 33 |
| SQ | 2 | 61 | 11 | 10 |
| SQ | 3 | 28 | 52 | 50 |
| SQ | 4 | 9 | 28 | 29 |
| SQ | 5 | 2 | 8 | 9 |
| SQ | 6 | 0 | 1 | 2 |
| SQ | 7+ | 0 | 0 | 0 |
| Distance [cm] | 0–0.5 | 14 | 0 | 0 |
| Distance [cm] | 0.5–1 | 22 | 4 | 3 |
| Distance [cm] | 1–2 | 22 | 20 | 19 |
| Distance [cm] | 2–3 | 14 | 21 | 21 |
| Distance [cm] | 3–4 | 9 | 20 | 20 |
| Distance [cm] | 4–6 | 9 | 20 | 20 |
| Distance [cm] | 6–8 | 4 | 8 | 8 |
| Distance [cm] | 8–12 | 3 | 6 | 6 |
| Distance [cm] | 12–20 | 1 | 1 | 2 |
| Distance [cm] | 20+ | 0 | 0 | 0 |

### Classifier: S/√B change vs background removed (per bin, quadrature)

| removed | 200–342 keV | 342–585 keV | 585–1000 keV | 1000–1710 keV | 1710–2924 keV | 2924–5000 keV | all bins |
|---|---|---|---|---|---|---|---|
| 50% | -17.5 | +0.4 | -1.4 | -3.6 | -3.0 | -2.2 | -8.2 |
| 70% | -28.2 | -9.0 | -7.4 | -9.9 | -9.1 | -8.0 | -17.1 |
| 80% | -41.7 | -15.7 | -12.1 | -14.0 | -13.8 | -14.2 | -25.8 |
| 90% | -54.2 | -28.5 | -19.4 | -23.6 | -24.8 | -24.7 | -36.8 |
| 95% | -65.2 | -39.9 | -33.9 | -38.5 | -36.9 | -37.3 | -48.7 |
| 98% | -74.4 | -58.6 | -51.9 | -52.0 | -51.0 | -55.2 | -63.0 |
