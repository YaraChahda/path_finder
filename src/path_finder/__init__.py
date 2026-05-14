"""
path_finder
───────────
Find the best retrosynthesis pathways for target molecules,
based on criteria selected by the user.

Developed by Yara Chahda, Corentin Postmann and Ines Ouchen — EPFL 2025.

After pip install path-finder-retrosynthesis:
    path-finder-setup    # first-time setup
    path-finder          # launch the app
"""
from __future__ import annotations

from path_finder.route_engine import find_best_routes

__version__ = "1.0.2"
__authors__ = ["Yara Chahda", "Corentin Portmann", "Ines Ouchen"]
