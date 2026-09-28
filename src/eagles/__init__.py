from .v2 import get_li_age # expose v2 get_li_age by default
from .v1 import get_li_age as get_li_age_v1 # but also allow v1

__all__ = ["get_li_age", "get_li_age_v1"]
