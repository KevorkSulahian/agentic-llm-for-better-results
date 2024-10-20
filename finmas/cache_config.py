from pathlib import Path

import diskcache

cache = diskcache.Cache(str(Path(__file__).parent.parent / ".finmas_cache"))
