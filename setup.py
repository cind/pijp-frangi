from setuptools import setup, find_packages
import pijp_frangi
setup(
    name="pijp-frangi",
    version=pijp_frangi.__version__,
    author="Serena Tang",
    author_email="serena.tang@ucsf.edu",
    packages=["pijp_frangi"],
    include_package_data=True,
    install_requires=[
        'pandas',
        'nibabel',
        'scipy',
        'numpy'
    ],
    entry_points="""
        [console_scripts]
        frangi=pijp_frangi:run
    """
)
