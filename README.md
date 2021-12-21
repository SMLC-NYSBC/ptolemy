# Ptolemy
A Python machine learning and computer vision library for automating cryo-EM data collection. The accompanying paper is available on [arxiv](https://arxiv.org/abs/2112.01534). Ptolemy is designed to handle localization and scoring of squares in low-mag images (TODO: resolution) and holes in medium-mag images (TODO: resolution). It works on both gold and carbon holey, untilted grids. 

<details><summary>Example Low Mag Image</summary><p>

  <img src="example_images/for_readme/lowmag.png" width="500">
  <img src="example_images/for_readme/lowmag_processed.png" width=500>
  
</p></details>

<details><summary>Example Med Mag Image</summary><p>

  <img src="example_images/for_readme/medmag.png" width="500">
  <img src="example_images/for_readme/medmag_processed.png" width=500>
  
</p></details>

## Functionality
Images and visualization are handled by the `Exposure` class in `ptolemy/images.py`, with algorithms for processing low and medium mag images in `ptolemy/algorithms.py`. The workflow is outlined in the tutorial notebooks. 

## Future
We plan to improve Ptolemy with active learning on individual data collection sessions, support for tilted grids, and superresolution (unbinned) medium-mag images.

## Dependencies
Tested with python 3.9

- pytorch
- torchvision
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib

## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
