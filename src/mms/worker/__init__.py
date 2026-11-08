"""The code that runs in a worker process (DESIGN.md §13.3), apart from mms: nothing here imports `mms`, and the top
level of each file uses only the standard library, so a foreign env's Python can run it with `python -I`."""
