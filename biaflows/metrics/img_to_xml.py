# A script for converting OME-TIFF labeled masks to the Particle Tracking Challenge format
# Authors: Martin Maska <xmaska@fi.muni.cz>, 2018
#          Romain Mormont <rmormont@uliege.be>, 2021

import numpy as np
from collections import defaultdict
from biaflows.helpers.util import imread


def img_to_tracks(fname):
    # Convert the tracking results saved in an OME-TIFF image to a dictionary of tracks
    img_data, order, _ = imread(fname, return_order=True)
    track_dict = defaultdict(list)
    where = np.where(img_data > 0)
    order_idx = {d: i for i, d in enumerate(order)}
    for val, point in zip(img_data[where], zip(*where)):
        track_dict[val].append([(point[order_idx.get(d, -1)] if d in order_idx else 0) for d in "TXYZ"])
    return track_dict


def tracks_to_xml(fname, track_dict, keep_labels):
    # Convert the dictionary of tracks to the XML format used in the Particle Tracking Challenge
    with open(fname, "w") as f:
        f.write('<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n')
        f.write('<root>\n')
        f.write('<TrackContestISBI2012 SNR=\"1\" density=\"low\" scenario=\"vesicle\">\n')

        for track in track_dict:
            if keep_labels: f.write('<particle>\n')
            for point in track_dict[track]:
                if not keep_labels: f.write('<particle>\n')
                f.write('<detection t=\"'+str(point[0])+'\" x=\"'+str(point[1])+'\" y=\"'+str(point[2])+'\" z=\"'+str(point[3])+'\"/>\n')
                if not keep_labels: f.write('</particle>\n')
            if keep_labels: f.write('</particle>\n')

        f.write('</TrackContestISBI2012>\n')
        f.write('</root>\n')
		
