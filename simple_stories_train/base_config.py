from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with standard settings for all configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
