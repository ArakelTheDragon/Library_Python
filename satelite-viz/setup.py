import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
    name="topgis-viz",
    version="0.0.1",
    author="ArakelTheDragon",
    author_email="ArakelTheDragon@gmail.com",
    description="Satelite images processing for smart cars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ciirc.cvut.cz/itsovyor/python.assignments/-/issues/2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
