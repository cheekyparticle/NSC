from setuptools import setup, find_packages

setup(
    name='NSC',  # Replace with your package's name
    version='1.0.0',  # Replace with your package's version
    packages=find_packages(include=["NSC", "NSC.*"]),  # Automatically find all packages in the directory
    description='A code for non-standard cosmology solving',  # Replace with your description
    long_description=open('README.md').read(),  # Include a detailed description from the README
    long_description_content_type='text/markdown',  # Markdown is supported in the setup metadata
    author='Andrew Cheek',  # Replace with your name
    author_email='acheek@sjtu.edu.cn',  # Replace with your email
    license='MIT',  # Replace with your chosen license
    url='https://github.com/cheekyparticle/NSC',  # Replace with your package link
)