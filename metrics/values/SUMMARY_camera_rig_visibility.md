# Camera Rig Headline Metrics

Values are medians across matched scenes. These ROI-based headline metrics evaluate camera-rig support for autonomous robot navigation through surface observability, redundancy, and free-space certainty, not reconstruction overlap.

|Rig|Cameras|Raw Depth MB/Sample|Observable Surface (%)|Redundant Surface (%)|Strong Redundancy (%)|Single-View Surface (%)|Blind Surface (%)|Navigation Visibility Score (%)|Stakeholder Interpretation|
|---|---|---|---|---|---|---|---|---|---|
|4CamClassic|4|4.9|90.0|62.7|9.9|26.6|10.0|59.6|Baseline reference; useful context for the original four-camera geometry.|
|3Cam|3|3.7|87.9|44.1|6.4|41.2|12.1|52.5|Cost-minimal option; lower data burden but high single-view fragility.|
|4CamAsym|4|4.9|92.8|69.8|29.2|23.6|7.2|69.3|Practical recommendation; large robustness gain over 3Cam for one added camera.|
|5Cam|5|6.1|95.8|79.2|53.8|15.5|4.2|78.5|Upper-bound robustness; strongest redundancy at highest data burden.|
