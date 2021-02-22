import os
import sys

import tifffile
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
from cytomine.models import Annotation, ImageInstance, ImageSequenceCollection, AnnotationCollection, Property
from cytomine.models.image import SliceInstanceCollection
from cytomine.models.track import Track, TrackCollection
from shapely.geometry import LineString
from sldc import DefaultTileBuilder, SemanticMerger

from biaflows.exporter.mask_to_points import mask_to_points_3d
from biaflows.helpers.util import BiaflowsSldcImage, imread, imwrite_ome
from biaflows.problemclass import *
from biaflows.exporter import mask_to_objects_2d, mask_to_objects_3d, AnnotationSlice, csv_to_points, \
    slices_to_mask, mask_to_points_2d, skeleton_mask_to_objects_2d, skeleton_mask_to_objects_3d, mask_to_objects_3dt
from shapely.affinity import affine_transform


DEFAULT_COLOR = "#FF0000"


def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def create_annotation_from_slice(_slice, id_image, image_height, id_project, label=None, upload_group_id=False):
    """
    Parameters
    ----------
    _slice: AnnotationSlice
    id_image: int
    image_height: int
    id_project: int
    label: int
    upload_group_id: bool

    Returns
    -------
    annotation: Annotation
        An annotation which is NOT saved
    """
    parameters = {
        "location": change_referential(_slice.polygon, image_height).wkt,
        "id_image": id_image,
        "id_project": id_project,
    }
    if upload_group_id:
        parameters["property"] = [{"key": "label", "value": _slice.label if label is None else label}]
    return Annotation(**parameters)


def get_depth_to_slice(image_instance, depth='auto'):
    """
    Parameters
    ----------
    image_instance: ImageInstance
        An image
    depth: str|tuple
        One of {'auto', 'time', 'zStack'} or a tuple containing {'time', 'zStack'} in any order. Which field to read for getting the depth.

    Returns
    -------
    depth2slice: dict
        If depth was a string, maps the depth number (time or zStack) to the SliceInstance.
        If depth was a tuple, maps the tuple containing depth numbers in the same order as the tuple mapping the sliceinstance
    """
    slices = SliceInstanceCollection().fetch_with_filter("imageinstance", image_instance.id)
    read_zstack = lambda s: s.zStack
    read_time = lambda s: s.time
    # map tuple to slice
    if isinstance(depth, tuple):
        return {
            tuple((read_time(slice) if d == "time" else read_zstack(slice)) for d in depth): slice
            for slice in slices
        }
    # map depth number to slice
    read_depth = read_zstack  # by default
    if depth is 'auto' and image_instance.duration > 1:
        read_depth = read_time
    elif depth is 'time':
        read_depth = read_time
    return {read_depth(slice): slice for slice in slices}


def create_track_from_slices(image, slices, depth2slice, id_project, track_prefix="object", label=None, upload_group_id=False, depth="time"):
    """Create an annotation track from a list of AnnotationSlice
    Parameters
    ----------
    image: ImageInstance
        The image instance in which the track is added
    slices: iterable (of AnnotationSlice)
        The polygon slices of the objects to draw
    depth2slice: dict
        A dictionary mapping the depths of the image instance with their respective SliceInstance
    id_project: int
        Project identifier
    track_prefix: str (default: "object")
        A prefix for the track name
    label: int|str (default: None)
        A label for the track
    upload_group_id: bool
        True to upload the group identifier
    depth: str
        Which depth field to read in the AnnotationSlice if both are present. One of {'time', 'depth'}.

    Returns
    -------
    saved_tracks: TrackCollection
        The saved track objects
    annotations: AnnotationCollection
        The annotations associated with the traped. The collection is NOT saved.
    """
    if label is None and len(slices) > 0:
        label = slices[0].label
    track = Track(name="{}-{}".format(track_prefix, label), id_image=image.id, color=None if upload_group_id else DEFAULT_COLOR).save()

    if upload_group_id:
        Property(track, key="label", value=label).save()

    collection = AnnotationCollection()
    for _slice in slices:
        collection.append(Annotation(
            location=change_referential(p=_slice.polygon, height=image.height).wkt,
            id_image=image.id,
            id_project=id_project,
            id_tracks=[track.id],
            slice=depth2slice[_slice.depth if _slice.time is None or depth == "depth" else _slice.time].id
        ))
    return track, collection


def create_tracking_from_slice_group(image, slices, slice2point, depth2slice, id_project, upload_object=False,
                                     track_prefix="object", label=None, upload_group_id=False):
    """Create a set of tracks and annotations to represent a tracked element. A trackline is created to reflect the
    movement of the object in the image. Optionally the object's polygon can also be uploaded.

    Parameters
    ----------
    image: ImageInstance
        An ImageInstance
    slices: list of AnnotationSlice
        A list of AnnotationSlice of one object
    slice2point: callable
        A function that transform a slice into its representative point to be used for generating the tracking line
    depth2slice: dict
        Maps time step with corresponding SliceInstance
    id_project: int
        Project identifier
    upload_object: bool
        True if the object should be uploaded as well (the trackline is uploaded in any case)
    track_prefix: str
        A prefix for the track name
    label: int (default: None)
        The label of the tracked object
    upload_group_id: bool
        True for uploading the object label with the track

    Returns
    -------
    saved_tracks: TrackCollection
        The saved track objects
    annotations: AnnotationCollection
        The annotations associated with the traped. The collection is NOT saved.
    """
    if label is None and len(slices) > 0:
        label = slices[0].label

    # create tracks
    tracks = TrackCollection()
    object_track = Track("{}-{}".format(track_prefix, label), image.id, color=None if upload_group_id else DEFAULT_COLOR).save()
    trackline_track = Track("{}-{}-trackline".format(track_prefix, label), image.id, color=None if upload_group_id else DEFAULT_COLOR).save()
    tracks.extend([object_track, trackline_track])

    if upload_group_id:
        Property(object_track, key="label", value=int(label)).save()
        Property(trackline_track, key="label", value=int(label)).save()

    # create actual annotations
    annotations = AnnotationCollection()
    sorted_group = sorted(slices, key=lambda s: s.time)
    prev_line = []
    for _slice in sorted_group:
        point = slice2point(_slice)
        if point.is_empty:  # skip empty points
            continue
        if len(prev_line) == 0 or not prev_line[-1].equals(point):
            prev_line.append(point)

        if len(prev_line) == 1:
            polygon = slice2point(_slice)
        else:
            polygon = LineString(prev_line)

        depth = _slice.time if _slice.depth is None else _slice.depth
        annotations.append(Annotation(
            location=change_referential(polygon, image.height).wkt,
            id_image=image.id,
            slice=depth2slice[depth].id,
            id_project=id_project,
            id_tracks=[trackline_track.id]
        ))

        if upload_object:
            annotations.append(Annotation(
                location=change_referential(_slice.polygon, image.height).wkt,
                id_image=image.id,
                slice=depth2slice[depth].id,
                id_project=id_project,
                id_tracks=[object_track.id]
            ))

    return tracks, annotations


def mask_convert(mask, image, project_id, mask_2d_fn, mask_3d_fn, track_prefix, upload_group_id=False):
    """Generic function to convert a mask into an annotation collection

    Parameters
    ----------
    mask: ndarray
    image: ImageInstance
    project_id: int
    mask_2d_fn: callable
    mask_3d_fn: callable
    track_prefix: str
    upload_group_id: bool

    Returns
    -------
    tracks: TrackCollection
        Tracks, which have been saved
    annotations: AnnotationCollection
        Annotation which have NOT been saved
    """
    tracks = TrackCollection()
    annotations = AnnotationCollection()
    if mask.ndim == 2:
        slices = mask_2d_fn(mask)
        annotations.extend([create_annotation_from_slice(
            s, image.id, image.height, project_id, upload_group_id=upload_group_id) for s in slices
        ])
    elif mask.ndim == 3:
        slices = mask_3d_fn(mask)
        depth_to_slice = get_depth_to_slice(image)
        for obj_id, obj in enumerate(slices):
            track, curr_annotations = create_track_from_slices(
                image, obj, label=obj_id, depth2slice=depth_to_slice,
                track_prefix=track_prefix, id_project=project_id,
                upload_group_id=upload_group_id
            )
            tracks.append(track)
            annotations.extend(curr_annotations)
    else:
        raise ValueError("Only supports 2D or 3D output images...")
    return tracks, annotations


def extract_annotations_objseg(out_path, in_image, project_id, track_prefix, upload_group_id=False, **kwargs):
    """
    Parameters
    ----------
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    track_prefix: str
    upload_group_id: bool
        True for uploading annotation group id
    kwargs: dict
    """
    image = in_image.object
    path = os.path.join(out_path, in_image.filename)
    data = imread(path)
    return mask_convert(
        data, image, project_id,
        mask_2d_fn=mask_to_objects_2d,
        mask_3d_fn=lambda m: mask_to_objects_3d(m, background=0, assume_unique_labels=True),
        track_prefix=track_prefix + "-object",
        upload_group_id=upload_group_id
    )


def extract_tiled_annotations(in_tiles, out_path, nj, label_merging=False):
    """
    in_images: iterable
        List of BiaflowsTile
    out_path: str
        Path of output tiles
    nj: BiaflowsJob
        A BIAflows job object
    label_merging: bool
        True for merging only polygons having the same label. False for merging based on geometry only
    """
    # regroup tiles by original images
    grouped_tiles = defaultdict(list)
    for in_tile in in_tiles:
        grouped_tiles[in_tile.in_image.original_filename].append(in_tile)

    default_tile_builder = DefaultTileBuilder()
    annotations = AnnotationCollection()
    for tiles in grouped_tiles.values():
        # recreate the topology
        in_image = tiles[0].in_image
        topology = BiaflowsSldcImage(in_image, is_2d=True).tile_topology(
            default_tile_builder,
            max_width=nj.flags["tile_width"],
            max_height=nj.flags["tile_height"],
            overlap=nj.flags["tile_overlap"])

        # extract polygons for each tile
        ids, polygons, labels = list(), list(), list()
        label = -1
        for tile in tiles:
            out_tile_path = os.path.join(out_path, tile.filename)
            slices = mask_to_objects_2d(imread(out_tile_path), offset=tile.tile.abs_offset[::-1])
            ids.append(tile.tile.identifier)
            polygons.append([s.polygon for s in slices])
            labels.append([s.label for s in slices])
            # save label for use after merging
            if len(slices) > 0:
                label = slices[0]

        # merge
        merged = SemanticMerger(tolerance=1).merge(ids, polygons, topology, labels=labels if label_merging else None)
        if label_merging:
            merged = merged[0]
        annotations.extend([create_annotation_from_slice(
            _slice=AnnotationSlice(p, label),
            id_image=in_image.object.id,
            image_height=in_image.object.height,
            id_project=nj.project.id,
        ) for p in merged])
    return annotations


def extract_annotations_pixcla(out_path, in_image, project_id, track_prefix, upload_group_id=False, **kwargs):
    """
    Parameters
    ----------
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    track_prefix: str
    upload_group_id: bool
        True for uploading annotation group id
    kwargs: dict
    """
    image = in_image.object
    path = os.path.join(out_path, in_image.filename)
    data = imread(path)
    return mask_convert(
        data, image, project_id,
        mask_2d_fn=mask_to_objects_2d,
        mask_3d_fn=lambda m: mask_to_objects_3d(m, background=0, assume_unique_labels=False),
        track_prefix=track_prefix + "-object",
        upload_group_id=upload_group_id
    )


def extract_annotations_objdet(out_path, in_image, project_id, track_prefix, is_csv=False, generate_mask=False,
                               result_file_suffix=".tif", has_headers=False, parse_fn=None, upload_group_id=False,
                               **kwargs):
    """
    Parameters:
    -----------
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    is_csv: bool
        True if the output data are stored in a csv file
    generate_mask: bool
        If result is in a CSV, True for generating a mask based on the points in the csv. Ignored if is_csv is False.
        The mask file is generated in out_path with the name "{in_image.id}.png".
    result_file_suffix: str
        Suffix of the result filename (prefix being the image id).
    has_headers: bool
        True if the csv contains some headers (ignored if is_csv is False)
    parse_fn: callable
        A function for extracting coordinates from the csv file (already separated) line.
    track_prefix: str
    upload_group_id: bool
        True for uploading annotation group id
    kwargs: dict
    """
    image = in_image.object
    file = str(image.id) + result_file_suffix
    path = os.path.join(out_path, file)

    tracks = TrackCollection()
    annotations = AnnotationCollection()
    if not os.path.isfile(path):
        print("No output file at '{}' for image with id:{}.".format(path, image.id), file=sys.stderr)
        return annotations

    # whether the points are stored in a csv or a mask
    if is_csv:
        if parse_fn is None:
            raise ValueError("parse_fn shouldn't be 'None' when result file is a CSV.")
        points = csv_to_points(path, has_headers=has_headers, parse_fn=parse_fn)
        annotations.extend([
            create_annotation_from_slice(point, image.id, image.height, project_id, upload_group_id=upload_group_id)
            for point in points
        ])

        if generate_mask:
            mask = slices_to_mask(points, imread(in_image.filepath).shape).squeeze()
            imwrite_ome(os.path.join(out_path, in_image.filename),
                        mask, SizeC=1, SizeX=mask.shape[-1], SizeY=mask.shape[-2], DimensionOrder="CXYZT")
    else:
        # points stored in a mask
        tracks, annotations = mask_convert(
            imread(path), image, project_id,
            mask_2d_fn=mask_to_points_2d,
            mask_3d_fn=lambda m: mask_to_points_3d(m, time=False, assume_unique_labels=False),
            track_prefix=track_prefix + "-object",
            upload_group_id=upload_group_id
        )

    return tracks, annotations


def extract_annotations_prttrk(out_path, in_image, project_id, track_prefix, upload_group_id=False, **kwargs):
    """
    Parameters:
    -----------
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    name_prefix: str
    upload_group_id: bool
    kwargs: dict
    """

    image = in_image.object
    path = os.path.join(out_path, in_image.filename)
    data = imread(path)

    if data.ndim != 3:
        raise ValueError("Annotation extraction for object tracking does not support masks with more than 3 dims...")

    slices = mask_to_points_3d(data, time=True, assume_unique_labels=True)
    time_to_image = get_depth_to_slice(image)

    tracks = TrackCollection()
    annotations = AnnotationCollection()
    for slice_group in slices:
        curr_tracks, curr_annots = create_tracking_from_slice_group(
            image, slice_group,
            slice2point=lambda _slice: _slice.polygon,
            depth2slice=time_to_image, id_project=project_id,
            upload_object=False, track_prefix=track_prefix + "-particle",
            upload_group_id=upload_group_id
        )
        tracks.extend(curr_tracks)
        annotations.extend(curr_annots)

    return tracks, annotations


def extract_annotations_objtrk(out_path, in_image, project_id, track_prefix, upload_group_id=False, **kwargs):
    """
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    track_prefix: str
    upload_group_id: bool
    kwargs: dict
    """
    image = in_image.object
    path = os.path.join(out_path, in_image.filename)
    data, dimorder, _ = imread(path, return_order=True)
    ndim = data.ndim - (1 if "C" in dimorder else 0)

    if ndim < 3:
        raise ValueError("Object tracking should be at least 3D (only {} spatial dimension(s) found)".format(ndim))

    tracks = TrackCollection()
    annotations = AnnotationCollection()

    if ndim == 3:
        slices = mask_to_objects_3d(data, time=True, assume_unique_labels=True)
        time_to_image = get_depth_to_slice(image)

        for slice_group in slices:
            curr_tracks, curr_annots = create_tracking_from_slice_group(
                image, slice_group,
                slice2point=lambda _slice: _slice.polygon.centroid,
                depth2slice=time_to_image, id_project=project_id,
                upload_object=True, upload_group_id=upload_group_id,
                track_prefix=track_prefix + "-object"
            )
            tracks.extend(curr_tracks)
            annotations.extend(curr_annots)
    elif ndim == 4:
        objects = mask_to_objects_3dt(mask=data)
        depths_to_image = get_depth_to_slice(image, depth=("time", "depth"))
        # TODO add tracking lines one way or another
        for time_steps in objects:
            label = time_steps[0][0].label
            track = Track(name="{}-{}".format(track_prefix, label), id_image=image.id,
                          color=None if upload_group_id else DEFAULT_COLOR).save()

            if upload_group_id:
                Property(track, key="label", value=label).save()

            annotations.extend([
                Annotation(
                    location=change_referential(p=slice.polygon, height=image.height).wkt,
                    id_image=image.id,
                    id_project=project_id,
                    id_tracks=[track.id],
                    slice=depths_to_image[(slice.time, slice.depth)].id
                ) for slices in time_steps for slice in slices
            ])

            tracks.append(track)

    else:
        raise ValueError("Annotation extraction for object tracking does not support masks with more than 4 dims...")

    return tracks, annotations


def extract_annotations_lootrc(out_path, in_image, project_id, track_prefix, upload_group_id=False, projection=0, **kwargs):
    """
    Parameters
    ----------
    out_path: str
    in_image: BiaflowsCytomineInput
    project_id: int
    track_prefix: str
    upload_group_id: bool
    projection: int
        Projection of the skeleton
    kwargs: dict
    """
    image = in_image.object
    path = os.path.join(out_path, in_image.filename)
    data = imread(path)
    tracks, collection = mask_convert(
        data, image, project_id,
        mask_2d_fn=skeleton_mask_to_objects_2d,
        mask_3d_fn=lambda m: skeleton_mask_to_objects_3d(m, background=0, assume_unique_labels=True, projection=projection),
        track_prefix=track_prefix + "-network",
        upload_group_id=upload_group_id
    )
    return tracks, collection


def upload_data(problemclass, nj, inputs, out_path, monitor_params=None, is_2d=True, **kwargs):
    """Upload annotations or any other related results to the server.

    Parameters
    ----------
    problemclass: str
        The problem class
    nj: CytomineJob|BiaflowsJob
        The CytomineJob or BiaflowsJob object. Ignored if do_export is True.
    inputs: list
        Input data as returned by the prepare_data
    out_path: str
        Output path
    monitor_params: dict|None
        A dictionnary of parameters to be passed to the data upload loop monitor.
    is_2d: bool
        True for 2D image, False for more than two dimensions.
    kwargs: dict
        Additional parameters for:
        * ObjDet/SptCnt: see function 'extract_annotations_objdet'
        * ObjSeg: see function 'extract_annotations_objseg'
    """
    if not nj.flags["do_upload_annotations"]:
        return
    if nj.flags["tiling"] and ((problemclass != CLASS_OBJSEG and problemclass != CLASS_PIXCLA) or not is_2d):
        print("Annot. upload is only supported for one of {ObjSeg, PixCla} in 2D when tiling is enabled.. skipping !")
        return
    if monitor_params is None:
        monitor_params = dict()

    annotations = AnnotationCollection()

    if nj.flags["tiling"]:
        annotations.extend(extract_tiled_annotations(inputs, out_path, nj, label_merging=problemclass == CLASS_PIXCLA))
    else:
        if problemclass == CLASS_OBJSEG:
            extract_fn = extract_annotations_objseg
        elif problemclass == CLASS_PIXCLA:
            extract_fn = extract_annotations_pixcla
        elif problemclass == CLASS_OBJDET or problemclass == CLASS_SPTCNT or problemclass == CLASS_LNDDET:
            extract_fn = extract_annotations_objdet
        elif problemclass == CLASS_LOOTRC or problemclass == CLASS_TRETRC:
            extract_fn = extract_annotations_lootrc
        elif problemclass == CLASS_PRTTRK:
            extract_fn = extract_annotations_prttrk
        elif problemclass == CLASS_OBJTRK:
            extract_fn = extract_annotations_objtrk
        else:
            raise NotImplementedError("Upload data does not support problem class '{}' yet.".format(problemclass))

        # whether or not to upload a unique identifier as a property with each detected object
        upload_group_id = not is_2d or problemclass in {CLASS_OBJTRK, CLASS_PRTTRK}

        tracks = TrackCollection()
        monitor_params["prefix"] = "Extract masks/points/... from output data"
        for in_image in nj.monitor(inputs, **monitor_params):
            curr_tracks, curr_annots = extract_fn(
                out_path, in_image, nj.project.id,
                track_prefix=str(nj.job.id),
                upload_group_id=upload_group_id,
                **kwargs
            )
            tracks.extend(curr_tracks)
            annotations.extend(curr_annots)

    nj.job.update(statusComment="Upload extracted annotations (total: {})".format(len(annotations)))
    annotations.save(chunk=20, n_workers=min(4, cpu_count() * 2))


