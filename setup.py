import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name='object_detection_poc',
    version='0.0.1',
    author="Tommy Yong",
    author_email="yongyct@gmail.com",
    description="Darknetflow POC",
    url="https://github.com/yongyct/darknetflow-poc",
    packages=setuptools.find_packages(),
    ext_modules=cythonize('object_detection_poc/utils/cython_utils/hello.pyx')
)
