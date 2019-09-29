import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name='darknetflow_poc',  
    version='0.0.1',
    author="Tommy Yong",
    author_email="yongyct@gmail.com",
    description="Darknetflow POC",
    url="https://github.com/yongyct/darknetflow-poc",
    packages=setuptools.find_packages(),
    ext_modules=cythonize('darknetflow_poc/utils/cython_utils/hello.pyx')
)
