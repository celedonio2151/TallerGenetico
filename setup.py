from distutils.core import setup
import pyinstaller

setup(
    name="My Game",
    version="1.0",
    description="A simple game made with Pygame",
    executables=["main.py"],
)
