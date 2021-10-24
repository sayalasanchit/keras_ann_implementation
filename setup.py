import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME="keras_ann_implementation"
USER_NAME="sayalasanchit"

setuptools.setup(
    name='src',
    version="0.0.2",
    author=USER_NAME,
    author_email="sayalasanchit@gmail.com",
    description="A small package for ANN implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    packages=['src'],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "PyYAML"
    ]
)