from setuptools import find_packages, setup
from typing import List

HUPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HUPEN_E_DOT in requirements:
            requirements.remove(HUPEN_E_DOT)
    
    return requirements

setup(
    name="boston_ml_project",
    version="0.0.1",
    author="Ayoub",
    author_email="ayoubbih119@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)