import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
    name="ciirc-utils",
    version="0.0.2",
    author="ArakelTheDragon",
    author_email="ArakelTheDragon@gmail.com",
    description="Common library functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
