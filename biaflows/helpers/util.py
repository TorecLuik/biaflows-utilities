import os
import tifffile
import numpy as np
from abc import ABCMeta, abstractmethod

import sldc
from cytomine.models._utilities import resolve_pattern


def get_ome_metadata(tiff):
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(tiff.ome_metadata)
    return list(list(tree)[0])[0].attrib


def imread_tifffile(image, np_dim_order="TZYXC", return_order=False):
    array = image.asarray().squeeze()
    if not image.is_ome or np_dim_order is None:
        # no metadata, so assume order 'TZYXC'
        return array
    metadata = get_ome_metadata(image)
    in_dim_order = metadata["DimensionOrder"][::-1]  # revert order so that it matches assarray
    dim_set = {"C", "X", "Y", "Z", "T"}
    non_empty_dim_set = {d for d in dim_set if int(metadata["Size{}".format(d)]) > 1}
    in_dim_order = "".join([d for d in in_dim_order if d in non_empty_dim_set])
    full_order = np_dim_order
    np_dim_order = "".join([d for d in np_dim_order if d in non_empty_dim_set])
    if in_dim_order == np_dim_order:
        final = array
    else:
        in_dim_idx = {d: i for i, d in enumerate(in_dim_order)}
        final = np.moveaxis(array, np.array([in_dim_idx[d] for d in np_dim_order]), np.arange(array.ndim))
    if return_order:
        return final, np_dim_order, full_order
    else:
        return final


def imread(filepath, np_dim_order="TZYXC", return_order=False):
    """Load image as an array and make sure dimensions are ordered as specified"""
    return imread_tifffile(tifffile.TiffFile(filepath), np_dim_order=np_dim_order, return_order=return_order)


def imwrite_ome(path, image, SizeC=1, SizeX=1, SizeY=1, SizeT=1, SizeZ=1, DimensionOrder="CXYZT", **metadata):
    meta = {
        **metadata,
        "SizeC": SizeC, "SizeX": SizeX, "SizeY": SizeY,
        "SizeT": SizeT, "SizeZ": SizeZ,
        "DimensionOrder": DimensionOrder
    }
    tifffile.imwrite(path, image, ome=True, metadata=meta)


def default_value(v, default):
    return default if v is None else v


def makedirs_ifnotexists(folder):
    """Create folder if not exists"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def check_field(d, f, target="dictionary"):
    if f not in d:
        raise ValueError("Missing field '{}' in {}".format(f, target))
    return d[f]


def split_filename(filename):
    return filename.rsplit(".", 1)


class BiaflowsInput(metaclass=ABCMeta):
    """A BIAflows input is a file that is used as input of a workflow (either ground truth or actual input).
    This class provides utilities methods for manipulating the input images
    instance.object allows to get the underlying input object
    instance.attached allows to get the list of files (as their filepath) attached to the input
    """
    def __init__(self, obj, attached=None, **kwargs):
        self._obj = obj
        self._attached = list() if attached is None else attached

    @property
    @abstractmethod
    def filepath(self):
        pass

    @property
    @abstractmethod
    def filename(self):
        """
        Returns
        -------
        str
        """
        pass

    @property
    def extension(self):
        return split_filename(self.filename)[1]

    @property
    def filename_no_extension(self):
        return split_filename(self.filename)[0]

    @property
    def object(self):
        return self._obj

    @property
    def attached(self):
        return self._attached


class BiaflowsCytomineInput(BiaflowsInput):
    def __init__(self, cytomine_model, in_path="", name_pattern="{id}.tif"):
        super().__init__(cytomine_model)
        self._in_path = in_path
        self._name_pattern = name_pattern

    @property
    def filename(self):
        return resolve_pattern(self._name_pattern, self.object)[0]

    @property
    def filepath(self):
        return os.path.join(self._in_path, self.filename)

    @property
    def original_filename(self):
        return getattr(self.object, self.filename_attribute)

    @property
    def filename_attribute(self):
        return "originalFilename"


class BiaflowsFilepath(BiaflowsInput):
    def __init__(self, filepath):
        super().__init__(filepath)

    @property
    def filepath(self):
        return self.object

    @property
    def filename(self):
        return os.path.basename(self.filepath)


class BiaflowsAttachedFile(BiaflowsCytomineInput):
    def __init__(self, attached_file, in_path="", name_pattern="{filename}"):  # change default pattern
        super().__init__(attached_file, in_path, name_pattern)

    @property
    def filename_attribute(self):
        return "filename"


# ----------------------------------------
# SLDC compatible Image classes for tiling
# ----------------------------------------

class BiaflowsSldcImage(sldc.Image):
    def __init__(self, in_image, is_2d=True):
        from .data_upload import imread
        self.in_image = in_image
        # currently a proof of concept, so load image in memory
        self.image = imread(in_image.filepath, is_2d)

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def channels(self):
        return self.image.shape[2]

    @property
    def np_image(self):
        return self.image


class BiaflowsTile(BiaflowsInput):
    def __init__(self, in_image, tile_path, tile):
        super(BiaflowsTile, self).__init__(in_image.object, in_image.attached)
        self.tile = tile
        self.in_image = in_image
        self.tile_path = tile_path

    @property
    def filepath(self):
        return os.path.join(self.tile_path, self.filename)

    @property
    def filename(self):
        extension = self.in_image.filename.rsplit(".", 1)[1]
        return "{}_{}-{}-{}-{}-{}.{}".format(
            self.in_image.filename.rsplit(".", 1)[0],
            self.tile.identifier,
            self.tile.abs_offset_y, self.tile.abs_offset_x,
            self.tile.height, self.tile.width, extension
        )
