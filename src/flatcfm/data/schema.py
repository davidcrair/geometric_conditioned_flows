"""condition schema helpers"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConditionFieldSchema:
    """condition field schema"""

    name: str
    source_column: str
    encoding: str = "categorical"


@dataclass(frozen=True)
class ConditionSchema:
    """condition schema"""

    perturbation: ConditionFieldSchema
    control_column: str
    control_value: Any
    perturbation_covariates: tuple[ConditionFieldSchema, ...] = ()
    sample_covariates: tuple[ConditionFieldSchema, ...] = ()
    output_obs_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "ConditionSchema":
        """build schema from config"""

        pert_cfg = cfg["perturbation"]
        pert_field = ConditionFieldSchema(
            name=pert_cfg.get("name", pert_cfg["source_column"]),
            source_column=pert_cfg["source_column"],
            encoding=pert_cfg.get("encoding", "categorical"),
        )
        pert_covs = tuple(
            ConditionFieldSchema(
                name=item.get("name", item["source_column"]),
                source_column=item["source_column"],
                encoding=item.get("encoding", "categorical"),
            )
            for item in cfg.get("perturbation_covariates", [])
        )
        sample_covs = tuple(
            ConditionFieldSchema(
                name=item.get("name", item["source_column"]),
                source_column=item["source_column"],
                encoding=item.get("encoding", "categorical"),
            )
            for item in cfg.get("sample_covariates", [])
        )
        return cls(
            perturbation=pert_field,
            control_column=cfg.get("control_column", "vehicle"),
            control_value=cfg.get("control_value", 1),
            perturbation_covariates=pert_covs,
            sample_covariates=sample_covs,
            output_obs_map=cfg.get("output_obs_map", {}),
        )

    @property
    def perturbation_key(self) -> str:
        """return perturbation key"""

        return self.perturbation.name

    @property
    def perturbation_source(self) -> str:
        """return perturbation source"""

        return self.perturbation.source_column

    @property
    def perturbation_covariate_names(self) -> tuple[str, ...]:
        """return perturbation covariate names"""

        return tuple(item.name for item in self.perturbation_covariates)

    @property
    def sample_covariate_names(self) -> tuple[str, ...]:
        """return sample covariate names"""

        return tuple(item.name for item in self.sample_covariates)

    def to_dict(self) -> dict:
        """serialize schema"""

        return {
            "perturbation": {
                "name": self.perturbation.name,
                "source_column": self.perturbation.source_column,
                "encoding": self.perturbation.encoding,
            },
            "control_column": self.control_column,
            "control_value": self.control_value,
            "perturbation_covariates": [
                {
                    "name": field.name,
                    "source_column": field.source_column,
                    "encoding": field.encoding,
                }
                for field in self.perturbation_covariates
            ],
            "sample_covariates": [
                {
                    "name": field.name,
                    "source_column": field.source_column,
                    "encoding": field.encoding,
                }
                for field in self.sample_covariates
            ],
            "output_obs_map": dict(self.output_obs_map),
        }
