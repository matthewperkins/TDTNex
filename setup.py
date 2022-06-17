import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TDTNex", # Replace with your own username
    version="0.2.3",
    author="Matthew Perkins",
    author_email="matthew.perkins@mssm.edu",
    description="Coordinate TDT data tanks with Nex Files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthewperkins/TDTNex",
    packages=setuptools.find_packages(),
    install_requires=['pandas',
                      'numpy',
                      'matplotlib',
                      'tdt',
                      'neo'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.*',
)
