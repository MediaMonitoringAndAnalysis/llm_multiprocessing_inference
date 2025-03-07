from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm_multiprocessing_inference",
    version="0.1.0",
    author="Reporter.ai (https://reporterai.org)",
    author_email="reporter.ai@boldcode.io",
    description="A package for parallel inference using OpenAI, other LLM APIs and local models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MediaMonitoringAndAnalysis/llm_multiprocessing_inference",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)