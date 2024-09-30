import yaml


CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r") as c:
    config = yaml.safe_load(c)

defaults = config.get("defaults", {})
