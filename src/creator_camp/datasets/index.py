from enum import Enum


# Github Release URL for datasets
# This will be removed after the contest ends due to the copyright issue.
base_git_path = "https://github.com/b-re-w/K-ICT_DataCreatorCamp_2025/releases/download/dt/"


class KompsatIndex(Enum):
    TRAIN = "TS_KS.zip"
    VALID = "VS_KS.zip"
    TRAIN_BBOX = "TL_KS_BBOX.zip"
    VALID_BBOX = "VL_KS_BBOX.zip"
    TRAIN_LINE = "TL_KS_LINE.zip"
    VALID_LINE = "VL_KS_LINE.zip"

    @property
    def url(self):
        return f"{base_git_path}{self.value}"


class SentinelIndex(Enum):
    TRAIN = "TS_SN10_SN10.zip"
    VALID = "VS_SN10_SN10.zip"
    TRAIN_MASK = "TL_SN10.zip"
    VALID_MASK = "VL_SN10.zip"
    TRAIN_GEMS = "TS_SN10_GEMS.zip"
    VALID_GEMS = "VS_SN10_GEMS.zip"
    TRAIN_AIR = "TS_SN10_AIR_POLLUTION.zip"
    VALID_AIR = "VS_SN10_AIR_POLLUTION.zip"

    @property
    def url(self):
        return f"{base_git_path}{self.value}"

    @property
    def urls(self):
        data_range = None
        match self:
            case SentinelIndex.TRAIN:
                data_range = range(1, 9)
            case SentinelIndex.TRAIN_GEMS:
                data_range = range(1, 3)
            case _:
                return [self.url]
        return [
            self.url.replace(".zip", f"_p{i}.zip") for i in data_range
        ]

    @property
    def names(self):
        data_range = None
        match self:
            case SentinelIndex.TRAIN:
                data_range = range(1, 9)
            case SentinelIndex.TRAIN_GEMS:
                data_range = range(1, 3)
            case _:
                return [self.value]
        return [
            self.value.replace(".zip", f"_p{i}.zip") for i in data_range
        ]


class LandsatIndex(Enum):
    TRAIN = "TS_LS30_LS30.zip"
    VALID = "VS_LS30_LS30.zip"
    TRAIN_MASK = "TL_LS30.zip"
    VALID_MASK = "VL_LS30.zip"

    @property
    def url(self):
        return f"{base_git_path}{self.value}"

    @property
    def urls(self):
        return [self.url]

    @property
    def names(self):
        return [self.value]
