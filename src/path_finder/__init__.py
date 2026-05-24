"""
path_finder
───────────
Find the best retrosynthesis pathways for target molecules,
based on criteria selected by the user.

Developed by Yara Chahda, Corentin Portmann and Ines Ouchen — EPFL 2026.

After pip install path-finder-retrosynthesis:
    path-finder-setup    # first-time setup
    path-finder          # launch the app
"""
from __future__ import annotations

from path_finder.route_engine import find_best_routes
from path_finder._about_ import __version__

__authors__ = ["Yara Chahda", "Corentin Portmann", "Ines Ouchen"]
