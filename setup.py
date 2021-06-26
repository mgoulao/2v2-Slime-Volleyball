from setuptools import setup

setup(
    name='multiagentslimevolleygym',
    version='0.0.1',
    keywords='games, environment, agent, rl, ai, gym',
    url='',
    description='2v2 Slime Volleyball Gym Environment',
    packages=['slimevolleygym'],
    install_requires=[
        'gym>=0.9.4',
        'numpy>=1.13.0',
        'opencv-python>=3.4.2.0'
    ]
)
