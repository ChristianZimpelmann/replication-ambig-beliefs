# import unittest
# from test_agent import TestAgent
# if __name__ == "__main__":
#     unittest.main(verbosity=1)
import sys

import pytest

if __name__ == "__main__":
    status = pytest.main([sys.argv[1]])
    sys.exit(status)
