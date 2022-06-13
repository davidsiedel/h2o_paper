from typing import Callable


class Load:
    function: Callable
    direction: int

    def __init__(self, function: Callable, direction: int):
        """

        Args:
            function:
            direction:
        """
        self.function = function
        self.direction = direction
        return
