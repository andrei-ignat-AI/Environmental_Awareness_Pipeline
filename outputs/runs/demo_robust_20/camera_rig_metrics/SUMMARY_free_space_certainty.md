# Free-Space Certainty Metrics

Values are medians across matched scenes using the active metric scope: `roi`. These metrics describe whether GT-empty ROI voxels are certified free by the depth cameras. Unknown free volume means not certified free; it is not treated as occupied geometry.

|Rig|Samples|Free Volume Voxels (Median)|Certified Free Volume (%)|Single-View Certified Free Volume (%)|Redundantly Certified Free Volume (%)|Unknown Free Volume (%)|
|---|---|---|---|---|---|---|
|4CamClassic|5|544053|88.0|28.0|60.1|12.0|
|3Cam|5|544053|90.1|29.4|60.3|9.9|
|4CamAsym|5|544053|91.7|19.5|72.2|8.3|
|5Cam|5|544053|92.4|16.2|76.4|7.6|
