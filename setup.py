from setuptools import setup, find_packages

with open('README.md', 'r') as file_readme:
    long_desc = file_readme.read()

setup(
    name='quantrl',
    version='0.0.7',
    author='Sampreet Kalita',
    author_email='sampreet.kalita@hotmail.com',
    desctiption='Quantum Control with Reinforcement Learning',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    keywords=['quantum', 'toolbox', 'reinforcement learning', 'python3'],
    url='https://github.com/sampreet/quantrl',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'
    ],
    license='BSD',
    install_requires=[
        'gymnasium',
        'matplotlib',
        'numpy<2.0.0',
        'scipy',
        'seaborn'
        'stable-baselines3',
        'tqdm'
    ],
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True
)
