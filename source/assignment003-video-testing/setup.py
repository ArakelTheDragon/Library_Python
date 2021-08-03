import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
    name="videotest",
    version="0.0.8",
    author="ArakelTheDragon",
    author_email="ArakelTheDragon@gmail.com",
    description="Basic video processing functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ciirc.cvut.cz/itsovyor/python.assignments/-/tree/master/source",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
