from setuptools import find_packages , setup
from typing import List

HYPER_E_DOT = '-e .'

def req_pakage(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)

    return requirements
setup(
    name='SensorpredictionTool',
    version='0.0.1',
    author='Swati Swapna',
    author_email='swatiswapnas.s@gmail.com',
    install_requires = req_pakage('requirements.txt'),
    packages=find_packages()
)
