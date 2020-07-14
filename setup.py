from setuptools import setup, find_packages

setup(
    name="pynn_genn",
    version="0.2.0",
    packages=find_packages(),

    # Metadata for PyPi
    url="https://github.com/genn-team/pynn_genn",
    author="University of Sussex",
    description="Tools for simulating neural models generated using PyNN 0.9 using "
                "the GeNN simulator",
    #long_description=replace_local_hyperlinks(read_file("README.rst")),
    license="GPLv2",
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",

        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Windows",

        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",

        "Topic :: Scientific/Engineering",
    ],
    keywords="GPU CUDA pynn neural simulation GeNN",

    # Requirements
    # **NOTE** PyNN really should be requiring lazyarray itself but it (0.9.2) doesn't seem to
    install_requires=["pynn>=0.9, <0.9.3", "pygenn >= 0.4.1", "lazyarray>=0.3, < 0.4",
                      "sentinel", "neo>=0.6, <0.7", "numpy>=1.10.0,!=1.16.*", "six"],
    zip_safe=False,  # Partly for performance reasons
)
