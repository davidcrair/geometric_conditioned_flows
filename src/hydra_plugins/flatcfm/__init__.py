"""hydra searchpath plugin so perturbench discovers flatcfm model configs"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class FlatCFMConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="flatcfm",
            path="pkg://flatcfm.perturbench.configs",
        )
