from setuptools import setup, find_packages


VERSION = "0.2.3"

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name="tf-inputs",
    version=VERSION,
    author="Daniel Watson",
    author_email="daniel.watson@nyu.edu",
    url="https://github.com/danielwatson6/tf-inputs.git",
    description="Input pipelines for TensorFlow that make sense.",
    long_description=LONG_DESCRIPTION,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    # install_requires=open("requirements.txt").read().split(),
)
