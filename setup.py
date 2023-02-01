import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastdist",
    version="1.1.5",
    author="tal boger",
    author_email="tboger10@gmail.com",
    description="Faster distance calculations in python using numba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talboger/fastdist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)