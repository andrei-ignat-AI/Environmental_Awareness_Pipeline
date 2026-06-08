# Layout Robustness

Layout-specific medians use the active metric scope: `roi`. The score is based on observability and redundancy, not reconstruction overlap.

|Layout|Rig|Samples|ROI Navigation Visibility Score (%)|Observable Surface (%)|Redundant Surface (%)|Strong Redundancy (%)|Blind Surface (%)|
|---|---|---|---|---|---|---|---|
|Cath baseline|4CamClassic|10|59.4|91.5|60.7|9.9|8.5|
|Cath baseline|3Cam|10|52.8|95.1|43.6|4.2|4.9|
|Cath baseline|4CamAsym|10|71.6|97.7|75.0|27.0|2.3|
|Cath baseline|5Cam|10|84.1|98.3|87.4|59.8|1.7|
|Cath shifted|4CamClassic|10|58.4|89.7|61.7|9.9|10.3|
|Cath shifted|3Cam|10|54.3|93.5|47.2|6.1|6.5|
|Cath shifted|4CamAsym|10|72.4|97.5|75.1|31.4|2.5|
|Cath shifted|5Cam|10|86.0|98.2|91.4|62.4|1.8|
|Neuro head-side|4CamClassic|10|61.5|92.3|67.9|10.1|7.7|
|Neuro head-side|3Cam|10|52.7|85.1|46.4|9.3|14.9|
|Neuro head-side|4CamAsym|10|68.0|89.9|66.3|35.0|10.1|
|Neuro head-side|5Cam|10|74.3|91.7|72.3|52.3|8.3|
|Neuro shifted|4CamClassic|10|59.8|87.7|64.1|9.3|12.3|
|Neuro shifted|3Cam|10|46.8|74.9|36.1|5.8|25.1|
|Neuro shifted|4CamAsym|10|63.1|87.6|59.0|26.1|12.4|
|Neuro shifted|5Cam|10|72.0|89.5|70.0|44.3|10.5|
