from setuptools import setup, find_packages

setup(
    name='wvbench',
    version='0.0.1',
    description='WildVision Benchmark',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/WildVision-AI/WildVision-Bench',
    install_requires=[
        "datasets",
        "fire",
        "matplotlib",
        "prettytable",
        "json5",
        "plotly",
        "fire",
        "PyYAML",
        "regex",
        "Pillow",
        "icecream",
        "ImageHash",
        "pandas",
        "plotly",
        "tiktoken",
        "prettytable",
        "scikit-learn",
    ]
)



# change it to pyproject.toml
# [build-system]