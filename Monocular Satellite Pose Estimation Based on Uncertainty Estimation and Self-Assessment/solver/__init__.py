"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .speed_solver import SpeedSolver


from typing import Dict

TASKS: Dict[str, BaseSolver] = {"detection": DetSolver, "landmarker": SpeedSolver}
