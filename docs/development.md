# Development

The project setup is inspired by both [Python for Data Science](https://www.python4data.science/en/latest/productive/index.html) and
the [Learn Scientific Python](https://learn.scientific-python.org/development/guides/style/) project. These projects give guidelines
to how to set up a research project that is reproducible and with good quality.

Commit messages are encouraged to follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

## Styling and pre-commit

To maintain the code quality when committing to the repo we use [pre-commit](https://pre-commit.com/) with
ruff, type checking for script files and formatting of pyproject.toml file. This ensures that these
code quality tools are run before any commit.

The configuration is stored in `.pre-commit-config.yaml`, and to set up the git hook scripts simply run
the following in the virtual environment:

```bash
pre-commit install
```

The pre-commits can be run on all files before committing by this command:

```bash
pre-commit run --all-files
```
