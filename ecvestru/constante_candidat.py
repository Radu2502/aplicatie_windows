"""Constantele candidatului (research/candidate.py) de care are nevoie aplicația, într-un singur loc.

Candidatul nu se modifică, iar pragurile detectorului de cai sunt scrise direct în apelul lui (nu ca nume), deci
aplicația le ia de aici, nu din copii scrise de mână în fiecare modul. tests/test_constante_candidat.py citește
research/candidate.py (cu ast, fără să-l importe sau să-l modifice) și cere exact aceleași valori. Fără Streamlit."""
from __future__ import annotations

MIN_VISIBILITY = 0.5        # sub ea, un punct MediaPipe nu intră în lanț (research/candidate.py: MIN_VISIBILITY)
HORSE_CATEGORY = "horse"    # detectorul de obiecte (EfficientDet-Lite0, COCO): category_allowlist
HORSE_SCORE = 0.3           # score_threshold
HORSE_RESULTS = 5           # max_results
