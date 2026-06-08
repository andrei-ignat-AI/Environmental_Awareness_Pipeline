# Free-Space Certainty Metrics

Values are medians across matched scenes using the active metric scope: `roi`. These metrics describe whether GT-empty ROI voxels are certified free by the depth cameras. Unknown free volume means not certified free; it is not treated as occupied geometry.

|Rig|Samples|Free Volume Voxels (Median)|Certified Free Volume (%)|Single-View Certified Free Volume (%)|Redundantly Certified Free Volume (%)|Unknown Free Volume (%)|
|---|---|---|---|---|---|---|
|4CamClassic|40|545234|87.7|28.3|59.3|12.3|
|3Cam|40|545234|90.0|31.5|59.1|10.0|
|4CamAsym|40|545234|92.6|20.3|72.3|7.4|
|5Cam|40|545234|93.6|15.3|78.3|6.4|
