from distutils.core import setup

import setuptools

setup(name='multi_visual_dash',
      version='1.0',
      description='The tool for visualization 2D/3D data',
      long_description='The tool for visualization 2D/3D data like plots, point clouds, maps, trajectories - '
                  'any figures which can be drawn by lines and points in 2D or 3D space',
      author='Ilya Zaychuk',
      author_email='zajchuk.ilya@gmail.com',
      url='https://github.com/GhostLate/MultiVisual-Dash',
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      install_requires=[
          'dash>=2.6.1',
          'websockets',
          'dash-extensions>=0.1.6',
          'plotly>=5.10.0',
          'kaleido>=0.2.1',
          'blosc2>=0.3.2',
      ],
      python_requires='>=3.8',
      )
