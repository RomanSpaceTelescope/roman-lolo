from setuptools import setup, find_packages

setup(
    name='roman-lolo',
    version='0.1.0',
    description='Roman WFI CRNL simulation and lamp-on/lamp-off analysis tools',
    author='Maxime Rizzo',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20',
        'astropy>=5.0',
        'matplotlib>=3.3',
        'photutils>=1.5',
        'scipy>=1.7',
        'pyds9>=1.8.1',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0',
            'pytest>=6.0',
        ],
    },
)
