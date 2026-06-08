# Camera Rig Visibility Summary

Values are medians across matched scenes. These ROI-based headline metrics evaluate camera-rig support for autonomous robot navigation through surface observability, redundancy, and free-space certainty, not reconstruction overlap.

|Rig|Cameras|Raw Depth MB/Sample|Observable Surface (%)|Redundant Surface (%)|Strong Redundancy (%)|Single-View Surface (%)|Blind Surface (%)|Navigation Visibility Score (%)|Stakeholder Interpretation|
|---|---|---|---|---|---|---|---|---|---|
|4CamClassic|4|4.9|89.9|66.1|10.4|24.8|10.1|61.0|Baseline reference; useful context for the original four-camera geometry.|
|3Cam|3|3.7|84.2|44.0|7.3|39.3|15.8|52.4|Cost-minimal option; lower data burden but high single-view fragility.|
|4CamAsym|4|4.9|89.9|65.4|28.9|23.6|10.1|67.7|Practical recommendation; large robustness gain over 3Cam for one added camera.|
|5Cam|5|6.1|90.8|72.4|50.5|18.4|9.2|73.9|Upper-bound robustness; strongest redundancy at highest data burden.|
